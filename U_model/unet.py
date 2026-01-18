from torch import nn
import scipy.stats as st
import torch
from .net_util import *
import numpy as np

import math
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def gauss_kernel(kernlen=21, nsig=3, channels=1):
    interval = (2 * nsig + 1.0) / (kernlen)
    x = np.linspace(-nsig - interval / 2.0, nsig + interval / 2.0, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kernel_raw = np.sqrt(np.outer(kern1d, kern1d))
    kernel = kernel_raw / kernel_raw.sum()
    out_filter = np.array(kernel, dtype=np.float32)
    out_filter = out_filter.reshape((1, 1, kernlen, kernlen))
    out_filter = np.repeat(out_filter, channels, axis=0)
    return out_filter


class EEC(nn.Module):
    def __init__(self, dim, bias=False):
        super(EEC, self).__init__()

        self.Conv = nn.Conv2d(2 * dim, dim, kernel_size=1, bias=bias)
        self.CA = ChannelAttention(dim)

    def forward(self, f_img, f_event, Mask):

        assert f_img.shape == f_event.shape, "the shape of image doesnt equal to event"
        b, c, h, w = f_img.shape

        F_event = f_event * Mask
        F_event = f_event + F_event
        F_cat = torch.cat([F_event, f_img], dim=1)
        F_conv = self.Conv(F_cat)
        w_c = self.CA(F_conv)
        F_event = F_event * w_c
        F_out = F_event + f_img

        return F_out


class ISC(nn.Module):
    def __init__(self, dim, num_heads=4, bias=False, LayerNorm_type="WithBias"):
        super(ISC, self).__init__()
        self.num_heads = num_heads
        self.norm = LayerNorm(dim, LayerNorm_type)
        self.SA = Spatio_Attention(dim, num_heads, bias)
        self.CA = ChannelAttention(dim)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(2 * dim, dim // 16, 1, bias=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(dim // 16, dim, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, f_img, f_event):

        assert f_img.shape == f_event.shape, "the shape of image doesnt equal to event"
        b, c, h, w = f_img.shape
        SA_att, V = self.SA(f_img)
        F_img = V @ SA_att
        F_img = rearrange(
            F_img, "b head c (h w) -> b (head c) h w", head=self.num_heads, h=h, w=w
        )
        CA_att = self.CA(f_img)
        F_img = F_img * CA_att
        F_img = F_img + f_img
        w_i = self.avg_pool(F_img)
        w_e = self.avg_pool(f_event)
        w = torch.cat([w_i, w_e], dim=1)
        w = self.fc2(self.relu1(self.fc1(w)))
        w = self.sigmoid(w)
        F_img = F_img * w
        F_event = f_event * (1 - w)
        F_event = F_event + F_img

        return F_event


def to_3d(x):
    return rearrange(x, "b c h w -> b (h w) c")


def to_4d(x, h, w):
    return rearrange(x, "b (h w) c -> b c h w", h=h, w=w)


class LayerNorm1(nn.Module):
    def __init__(self, dim):
        super(LayerNorm1, self).__init__()

        self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class EDFFN(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(EDFFN, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.patch_size = 8

        self.dim = dim
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv = nn.Conv2d(
            hidden_features * 2,
            hidden_features * 2,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=hidden_features * 2,
            bias=bias,
        )

        self.fft = nn.Parameter(
            torch.ones((dim, 1, 1, self.patch_size, self.patch_size // 2 + 1))
        )
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        x = self.project_out(x)

        x_patch = rearrange(
            x,
            "b c (h patch1) (w patch2) -> b c h w patch1 patch2",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )
        x_patch_fft = torch.fft.rfft2(x_patch.float())
        x_patch_fft = x_patch_fft * self.fft
        x_patch = torch.fft.irfft2(x_patch_fft, s=(self.patch_size, self.patch_size))
        x = rearrange(
            x_patch,
            "b c h w patch1 patch2 -> b c (h patch1) (w patch2)",
            patch1=self.patch_size,
            patch2=self.patch_size,
        )

        return x

class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=8,
        d_conv=3,
        expand=2.0,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.0,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(
            self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs
        )
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.GELU()

        self.x_proj = (
            nn.Linear(
                self.d_inner,
                (self.dt_rank + self.d_state * 2),
                bias=False,
                **factory_kwargs,
            ),
        )
        self.x_proj_weight = nn.Parameter(
            torch.stack([t.weight for t in self.x_proj], dim=0)
        )  # (K=4, N, inner)
        del self.x_proj

        self.x_conv = nn.Conv1d(
            in_channels=(self.dt_rank + self.d_state * 2),
            out_channels=(self.dt_rank + self.d_state * 2),
            kernel_size=7,
            padding=3,
            groups=(self.dt_rank + self.d_state * 2),
        )

        self.dt_projs = (
            self.dt_init(
                self.dt_rank,
                self.d_inner,
                dt_scale,
                dt_init,
                dt_min,
                dt_max,
                dt_init_floor,
                **factory_kwargs,
            ),
        )
        self.dt_projs_weight = nn.Parameter(
            torch.stack([t.weight for t in self.dt_projs], dim=0)
        )  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(
            torch.stack([t.bias for t in self.dt_projs], dim=0)
        )  # (K=4, inner)
        del self.dt_projs

        self.A_logs = self.A_log_init(
            self.d_state, self.d_inner, copies=1, merge=True
        )  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=1, merge=True)  # (K=4, D, N)

        self.selective_scan = selective_scan_fn

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(
            self.d_inner, self.d_model, bias=bias, **factory_kwargs
        )
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else None

    @staticmethod
    def dt_init(
        dt_rank,
        d_inner,
        dt_scale=1.0,
        dt_init="random",
        dt_min=0.001,
        dt_max=0.1,
        dt_init_floor=1e-4,
        **factory_kwargs,
    ):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs)
            * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_core(self, x: torch.Tensor):
        B, C, H, W = x.shape
        L = H * W
        K = 1
        x_hwwh = x.view(B, 1, -1, L)
        xs = x_hwwh

        x_dbl = torch.einsum(
            "b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight
        )
        x_dbl = self.x_conv(x_dbl.squeeze(1)).unsqueeze(1)

        dts, Bs, Cs = torch.split(
            x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2
        )
        dts = torch.einsum(
            "b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight
        )
        xs = xs.float().view(B, -1, L)
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  # (k * d)
        # print(As.shape, Bs.shape, Cs.shape, Ds.shape, dts.shape)

        out_y = self.selective_scan(
            xs,
            dts,
            As,
            Bs,
            Cs,
            Ds,
            z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        return out_y[:, 0]

    def forward(self, x: torch.Tensor, **kwargs):
        x = rearrange(x, "b c h w -> b h w c")
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x))
        y1 = self.forward_core(x)
        assert y1.dtype == torch.float32
        y = y1
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.gelu(z)
        out = self.out_proj(y)
        out = rearrange(out, "b h w c -> b c h w")

        return out

# EVSSM
class EVS(nn.Module):
    def __init__(
        self,
        dim,
        ffn_expansion_factor=3,
        bias=False,
        LayerNorm_type="WithBias",
        att=False,
        idx=3,
        patch=128,
    ):
        super(EVS, self).__init__()

        self.att = att
        self.idx = idx
        if self.att:
            self.norm1 = LayerNorm1(dim)
            self.attn = SS2D(d_model=dim, patch=patch)

        self.norm2 = LayerNorm1(dim)
        self.ffn = EDFFN(dim, ffn_expansion_factor, bias)

        self.kernel_size = (patch, patch)

    def forward(self, x):
        if self.att:

            if self.idx % 2 == 1:
                x = torch.flip(x, dims=(-2, -1)).contiguous()
            if self.idx % 2 == 0:
                x = torch.transpose(x, dim0=-2, dim1=-1).contiguous()

            x = x + self.attn(self.norm1(x))

        x = x + self.ffn(self.norm2(x))

        return x


class EN_Block0812(nn.Module):

    def __init__(
        self, in_channels, out_channels, BIN, kernel_size=3, reduction=4, bias=False
    ):
        super(EN_Block0812, self).__init__()
        self.BIN = BIN
        act = nn.ReLU(inplace=True)
        self.conv = conv(in_channels, out_channels, 3, bias=bias)
        self.EVSs = [EVS(out_channels, att=True) for _ in range(BIN)]
        self.EVSs = nn.Sequential(*self.EVSs)

    def forward(self, x):
        x = self.conv(x)
        x = self.EVSs(x)
        return x


class Restoration(nn.Module):
    """Modified version of Unet from SuperSloMo."""

    def __init__(
        self, inChannels_img, inChannels_event, outChannels, args, ends_with_relu=False
    ):
        super(Restoration, self).__init__()
        self._ends_with_relu = ends_with_relu
        self.num_heads = 4
        self.act = nn.ReLU(inplace=True)

        self.channels = [64, 64, 128, 256]

        self.encoder_img_1 = EN_Block0812(inChannels_img, self.channels[0], 2)
        self.encoder_img_2 = EN_Block0812(self.channels[0], self.channels[1], 2)
        self.encoder_img_3 = EN_Block0812(self.channels[1], self.channels[2], 1)
        self.encoder_img_4 = EN_Block0812(self.channels[2], self.channels[3], 1)

        self.encoder_event_1 = EN_Block0812(inChannels_event, self.channels[0], 2)
        self.encoder_event_2 = EN_Block0812(self.channels[0], self.channels[1], 2)
        self.encoder_event_3 = EN_Block0812(self.channels[1], self.channels[2], 1)
        self.encoder_event_4 = EN_Block0812(self.channels[2], self.channels[3], 1)

        self.down = DownSample()

        self.EEC_1 = EEC(self.channels[0])
        self.EEC_2 = EEC(self.channels[1])

        self.ISC_3 = ISC(self.channels[2])
        self.ISC_4 = ISC(self.channels[3])

        self.decoder_img_1 = DE_Block(self.channels[3], self.channels[2])
        self.decoder_img_2 = DE_Block(self.channels[2], self.channels[1])
        self.decoder_img_3 = DE_Block(self.channels[1], self.channels[0])

        self.decoder_event_1 = DE_Block(self.channels[3], self.channels[2])
        self.decoder_event_2 = DE_Block(self.channels[2], self.channels[1])
        self.decoder_event_3 = DE_Block(self.channels[1], self.channels[0])

        self.weight_fusion_1 = Weight_Fusion(self.channels[2])
        self.weight_fusion_2 = Weight_Fusion(self.channels[1])
        self.weight_fusion_3 = Weight_Fusion(self.channels[0])

        self.out = nn.Conv2d(self.channels[0], outChannels, 3, stride=1, padding=1)

    def blur(self, x, kernel=21, channels=3, stride=1, padding="same"):
        kernel_var = (
            torch.from_numpy(gauss_kernel(kernel, 3, channels)).to(device).float()
        )
        return torch.nn.functional.conv2d(
            x, kernel_var, stride=stride, padding=int((kernel - 1) / 2), groups=channels
        )

    def forward(self, input_img, input_event):

        M0 = torch.clamp(
            self.blur(
                self.blur(
                    torch.sum(torch.abs(input_event), axis=1, keepdim=True),
                    kernel=7,
                    channels=1,
                ),
                kernel=7,
                channels=1,
            ),
            0,
            1,
        )

        img_1 = self.encoder_img_1(input_img)
        event_1 = self.encoder_event_1(input_event)

        down_img_1 = self.down(img_1)
        down_event_1 = self.down(event_1)
        M1 = self.blur(M0, kernel=5, channels=1, padding=2, stride=2)
        fuse_img_1 = self.EEC_1(down_img_1, down_event_1, M1)

        img_2 = self.encoder_img_2(fuse_img_1)
        event_2 = self.encoder_event_2(down_event_1)

        down_img_2 = self.down(img_2)
        down_event_2 = self.down(event_2)
        M2 = self.blur(M1, kernel=5, channels=1, padding=2, stride=2)
        fuse_img_2 = self.EEC_2(down_img_2, down_event_2, M2)

        img_3 = self.encoder_img_3(fuse_img_2)
        event_3 = self.encoder_event_3(down_event_2)

        down_img_3 = self.down(img_3)
        down_event_3 = self.down(event_3)
        fuse_event_3 = self.ISC_3(down_img_3, down_event_3)

        img_4 = self.encoder_img_4(down_img_3)
        event_4 = self.encoder_event_4(fuse_event_3)
        event_4 = self.ISC_4(img_4, event_4)

        up_img_1 = self.decoder_img_1(img_4, img_3)
        up_event_1 = self.decoder_event_1(event_4, event_3)

        fuse_img_1 = self.weight_fusion_1(up_img_1, up_event_1)

        up_img_2 = self.decoder_img_2(fuse_img_1, img_2)
        up_event_2 = self.decoder_event_2(up_event_1, event_2)

        fuse_img_2 = self.weight_fusion_2(up_img_2, up_event_2)

        up_img_3 = self.decoder_img_3(fuse_img_2, img_1)
        up_event_3 = self.decoder_event_3(up_event_2, event_1)

        de_fuse = self.weight_fusion_3(up_img_3, up_event_3)

        out = self.out(de_fuse)

        out = out + input_img
        return out

if __name__ == "__main__":
    from torchinfo import summary

    net = Restoration(3, 6, 3, None).cuda()
    summary(
        net,
        input_size=[(1, 3, 256, 256), (1, 6, 256, 256)],
        col_names=["input_size", "output_size", "num_params"],
        verbose=1,
    )
