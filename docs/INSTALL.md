<img src="../docs/figs/logo2.png" align="right" width="34%">

# Installation

### Setup Environment

This codebase is tested with `torch==1.11.0` and `torchvision==0.12.0`, with `CUDA 11.3` and `gcc 7.3.0`. In order to successfully reproduce the results reported in our paper, we recommend you to follow the exact same versions. However, similar versions that came out lately should be good as well.
```shell
conda create -n robodepth python=3.10
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
pip3 install opencv-python==4.6.0.66 timm==0.6.7 tensorboardX scikit-image==0.19.2 matplotlib yacs
```
```shell
pip install dotmap wandb einops tqdm
```

To create common corruptions, install the following packages into the environment:
```shell
pip install imagecorruptions
```

For the [Monocular Depth Estimation Toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox), install the following into the environment:
```shell
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.11/index.html
cd zoo/Monocular-Depth-Estimation-Toolbox/
pip3 install -e .
```
