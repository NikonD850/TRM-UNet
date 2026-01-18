# [ICASSP'26] TRM-UNet

## Installation

The model is built in PyTorch 2.5.1+cu124 and tested on Ubuntu 20.04 environment (Python 3.10, nvcc 12.8, mamba_ssm 2.2.2).

Install warmup scheduler
    cd pytorch-gradual-warmup-lr; python setup.py install; cd ..

## Training and Evaluation
### Train
- Download the [GoPro events train/test dataset](https://pan.baidu.com/s/1UKV-sPGo9mRf7XJjZDoF7Q) (code: kmaz) to your data root (provided by AHDINet's author)
- Change both training.yml and config.py to your settings.
- Train the model with default arguments by running
  
  python main_train.py

### Evaluation
- Download the [GoPro events test dataset](https://pan.baidu.com/s/1UKV-sPGo9mRf7XJjZDoF7Q) (code: kmaz) to ./Datasets
- Download the pretrained model TODO
- Test the model with default arguments by running

  python main_test.py

## Acknowledgement
Thanks to the inspirations and codes from [AHDINet](https://github.com/wyang-vis/AHDINet) and [EVSSM](https://github.com/kkkls/EVSSM/)
