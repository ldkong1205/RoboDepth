<img src="../figs/logo2.png" align="right" width="34%">

# Installation

### Setup Enviroment

This codebase is tested with `torch==1.11.0` and `torchvision==0.12.0`. In order to successfully reproduce the results reported in our paper, we recommend you to follow the exact same versions.
```shell
conda create -n robodepth python=3.9
conda activate robodepth
```
```shell
# CUDA 10.2
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=10.2 -c pytorch

# CUDA 11.3
conda install pytorch==1.11.0 torchvision==0.12.0 cudatoolkit=11.3 -c pytorch

# CPU Only
conda install pytorch==1.11.0 torchvision==0.12.0 cpuonly -c pytorch
```

### Install Packages

Please install the following packages into the environment:
```shell
pip3 install opencv-python==4.6.0.66
pip3 install timm==0.6.7
pip3 install tensorboardX
pip3 install imagecorruptions
pip3 install scikit-image==0.19.2
pip3 install matplotlib
pip3 install yacs
```
For the [Monocular Depth Estimation Toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox), install the following:
```shell
pip3 install mmcv-full==1.3.13
cd zoo/Monocular-Depth-Estimation-Toolbox/
pip3 install -e .
```
