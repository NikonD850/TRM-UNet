# [ICASSP'26] [TRM-UNet: An Efficient Event-Guided Motion Deblurring Network](https://xplorestaging.ieee.org/document/11462796)

[News] You may also want to check our new updates:

- **Event Deblur Pro (2026.03)** [Code](https://github.com/NikonD850/NTIRE26_event_deblur) 🏆 3rd Place of 2nd Event-based Image Deblurring Challenge

- **REGDSSM (2026.03)** [Code](https://github.com/NikonD850/REGDSSM)  🚩 Accepted by **ICME'26**

TL;DR: New state-of-the-art performance on the GoPro dataset while using only 64.1% of the parameters and 78.3% of the FLOPs compared to leading method AHDINet.

![Comparison](cmp.png)

## Installation
```
git clone https://github.com/NikonD850/TRM-UNet.git
cd TRM-UNet

# Create Conda Env
conda create -n trm-unet python=3.10 -y
conda activate trm-unet
pip install torch==2.5.1+cu124 torchvision==0.20.1+cu124 --index-url https://download.pytorch.org/whl/cu124

# Install Mamba
git clone https://github.com/state-spaces/mamba.git
cd mamba
git checkout 8ffd905
python -m pip install . --no-build-isolation
cd ..

# Install other dependences
python -m pip install matplotlib scikit-image opencv-python yacs joblib natsort h5py tqdm timm thop

# Install warm-up
cd pytorch-gradual-warmup-lr
python setup.py install
cd ..
```

## Training and Evaluation
The model is trained with 4 NVIDIA RTX 3090 24G, Ubuntu 20.04, NVIDIA Driver 570.86.10 and nvcc 12.8 .

The time for 1 epoch (1000 iterations) is within 55 minutes, including both training and validating.

For **TRAINING SPEED UP**, please follow our [new repository](https://github.com/NikonD850/NTIRE26_event_deblur).
### Train
- Download the [GoPro events train/test dataset](https://pan.baidu.com/s/1UKV-sPGo9mRf7XJjZDoF7Q) (code: kmaz) to your data root (provided by AHDINet's authors)
- Change both training.yml and config.py to your settings.
- Train the model with default arguments by running

```
 nohup python main_train.py > TRM-UNet-train.log 2>&1 &
```
### Evaluation
- Download the [GoPro events test dataset](https://pan.baidu.com/s/1UKV-sPGo9mRf7XJjZDoF7Q) (code: kmaz) to your data root (provided by AHDINet's authors)
- Download the [pretrained model](https://drive.google.com/file/d/1NAVTfdbPVsU1MIbuRhS0V4C6BMPzAecJ/view?usp=sharing) to TRM-UNet-512/models/TRM-UNet/model_best.pth
- Change both testing.yml and config.py to your settings.
- Test the model with default arguments by running

```
  python main_test.py
```
## Acknowledgement
Thanks to the inspirations and codes from [AHDINet](https://github.com/wyang-vis/AHDINet) and [EVSSM](https://github.com/kkkls/EVSSM/)

## Cite this work (BibTeX)
```
@INPROCEEDINGS{11462796,
  author={Fan, Dawei and Tang, Xiongxin and Chen, Qiao and Xu, Fanjiang},
  booktitle={ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={TRM-UNet: An Efficient Event-Guided Motion Deblurring Network}, 
  year={2026},
  pages={9047-9051},
  doi={10.1109/ICASSP55912.2026.11462796}}
```
