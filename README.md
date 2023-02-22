<p align="right">English | <a href="">简体中文</a></p>

<p align="center">
  <img src="docs/figs/logo.png" align="center" width="32%">
  
  <h3 align="center"><strong>RoboDepth: Robust Out-of-Distribution Depth Estimation under Corruptions</strong></h3>

  <p align="center">
    <a href="https://scholar.google.com/citations?user=-j1j7TkAAAAJ">Lingdong Kong</a><sup>1</sup>&nbsp;
    <a href="https://github.com/Daniel-xsy">Shaoyuan Xie</a><sup>2</sup>&nbsp;
    <a href="https://scholar.google.com/citations?user=UyooQDYAAAAJ">Hanjiang Hu</a><sup>3</sup>&nbsp;
    <a href="https://scholar.google.com/citations?user=9I7uKooAAAAJ">Benoit Cottereau</a><sup>4</sup>&nbsp;
    <a href="">Lai Xing Ng</a><sup>5</sup>&nbsp;
    <a href="https://scholar.google.com/citations?user=nFP2ldkAAAAJ">Wei Tsang Ooi</a><sup>1</sup>&nbsp;
    <br>
    <sup>1</sup>National University of Singapore&nbsp;&nbsp;
    <sup>2</sup>Huazhong Univerisity of Science and Technology<br>
    <sup>3</sup>Carnegie Mellon University&nbsp;&nbsp;
    <sup>4</sup>CNRS&nbsp;&nbsp;
    <sup>5</sup>A*STAR
  </p>

</p>

<p align="center">
  <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/Paper-%F0%9F%93%83-blue">
  </a>
  
  <a href="https://ldkong.com/RoboDepth" target='_blank'>
    <img src="https://img.shields.io/badge/Project-%F0%9F%94%97-lightblue">
  </a>
  
  <a href="https://huggingface.co/spaces/ldkong/RoboDepth" target='_blank'>
    <img src="https://img.shields.io/badge/Demo-%F0%9F%8E%AC-pink">
  </a>
  
  <a href="https://zhuanlan.zhihu.com/p/592479725" target='_blank'>
    <img src="https://img.shields.io/badge/%E4%B8%AD%E8%AF%91%E7%89%88-%F0%9F%90%BC-red">
  </a>
</p>


## About
**RoboDepth** is a comprehensive evaluation benchmark designed for probing the **robustness** of monocular depth estimation algorithms. It includes **18 common corruption** types, ranging from weather and lighting conditions, sensor failure and movement, and noises during data processing.

<p align="center">
  <img src="docs/figs/taxonomy.png" align="center" width="95%">
</p>


## Updates
- [2023.01] - The `NYUDepth2-C` dataset is ready to be downloaded! See [here](docs/DATA_PREPARE.md) for more details.
- [2023.01] - Evaluation server for Track 2 (fully-supervised depth estimation) is available on [this](https://codalab.lisn.upsaclay.fr/competitions/9821) page.
- [2023.01] - Evaluation server for Track 1 (self-supervised depth estimation) is available on [this](https://codalab.lisn.upsaclay.fr/competitions/9418) page.
- [2022.11] - We are organizing the [1st RoboDepth Competition](https://robodepth.github.io/) at [ICRA 2023](https://www.icra2023.org/). Join the challenge today! :raising_hand:
- [2022.11] - The `KITTI-C` dataset is ready to be downloaded! See [here](docs/DATA_PREPARE.md) for more details.


## Outline
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Getting Started](#getting-started)
- [Model Zoo](#model-zoo)
- [Create Corruption Sets](#create-corruption-sets)
- [TODO List](#todo-list)
- [Citation](#citation)
- [License](#license)
- [Acknowledgements](#acknowledgements)


## Installation
Kindly refer to [INSTALL.md](docs/INSTALL.md) for the installation details.


## Data Preparation

### RoboDepth Benchmark
Kindly refer to [DATA_PREPARE.md](docs/DATA_PREPARE.md) for the details to prepare the <sup>1</sup>[KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), <sup>2</sup>[KITTI-C](), <sup>3</sup>[Cityscapes](), <sup>4</sup>[NYUDepth2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html), and <sup>5</sup>[NYUDepth2-C]() datasets.

### Competition @ ICRA 2023
Kindly refer to [this](https://github.com/ldkong1205/RoboDepth/blob/main/competition/README.md) page for the details to prepare the training and evaluation data associated with the [1st RoboDepth Competition](https://robodepth.github.io/) at the 40th IEEE Conference on Robotics and Automation ([ICRA 2023](https://www.icra2023.org/)).


## Getting Started
Kindly refer to [GET_STARTED.md](docs/GET_STARTED.md) to learn more usage about this codebase.


## Model Zoo

### :oncoming_automobile: - Outdoor Depth Estimation
<details open>
<summary>&nbsp<b>Self-Supervised Depth Estimation</b></summary>

> - [x] **[MonoDepth2](https://arxiv.org/abs/1806.01260), ICCV 2019.** <sup>[**`[Code]`**](https://github.com/nianticlabs/monodepth2)</sup>
> - [x] **[DepthHints](https://arxiv.org/abs/1909.09051), ICCV 2019.** <sup>[**`[Code]`**](https://github.com/nianticlabs/depth-hints)</sup>
> - [x] **[MaskOcc](https://arxiv.org/abs/1908.11112), arXiv 2019.** <sup>[**`[Code]`**](https://github.com/schelv/monodepth2)</sup>
> - [x] **[DNet](https://arxiv.org/abs/2004.05560), IROS 2020.** <sup>[**`[Code]`**](https://github.com/TJ-IPLab/DNet)</sup>
> - [x] **[CADepth](https://arxiv.org/abs/2112.13047), 3DV 2021.** <sup>[**`[Code]`**](https://github.com/kamiLight/CADepth-master)</sup>
> - [ ] **[TC-Depth](https://arxiv.org/abs/2110.08192), 3DV 2021.** <sup>[**`[Code]`**](https://github.com/DaoyiG/TC-Depth)</sup>
> - [x] **[HR-Depth](https://arxiv.org/abs/2012.07356), AAAI 2021.** <sup>[**`[Code]`**](https://github.com/shawLyu/HR-Depth)</sup>
> - [ ] **[Insta-DM](https://arxiv.org/abs/2102.02629), AAAI 2021.** <sup>[**`[Code]`**](https://github.com/SeokjuLee/Insta-DM)</sup>
> - [x] **[DIFFNet](https://arxiv.org/pdf/2110.09482.pdf), BMVC 2021.** <sup>[**`[Code]`**](https://github.com/brandleyzhou/DIFFNet)</sup>
> - [x] **[ManyDepth](https://arxiv.org/abs/2104.14540), CVPR 2021.** <sup>[**`[Code]`**](https://github.com/nianticlabs/manydepth)</sup>
> - [ ] **[EPCDepth](https://arxiv.org/abs/2109.12484), ICCV 2021.** <sup>[**`[Code]`**](https://github.com/prstrive/EPCDepth)</sup>
> - [x] **[FSRE-Depth](http://arxiv.org/abs/2108.08829), ICCV 2021.** <sup>[**`[Code]`**](https://github.com/hyBlue/FSRE-Depth)</sup>
> - [ ] **[DepthFormer](https://arxiv.org/abs/2204.07616), CVPR 2022.** <sup>[**`[Code]`**](https://github.com/TRI-ML/vidar)</sup>
> - [ ] **[DynaDepth](https://arxiv.org/abs/2207.04680), ECCV 2022.** <sup>[**`[Code]`**](https://github.com/SenZHANG-GitHub/ekf-imu-depth)</sup>
> - [ ] **[DynamicDepth](https://arxiv.org/abs/2203.15174), ECCV 2022.** <sup>[**`[Code]`**](https://github.com/AutoAILab/DynamicDepth)</sup>
> - [ ] **[RA-Depth](https://arxiv.org/abs/2207.11984), ECCV 2022.** <sup>[**`[Code]`**](https://github.com/hmhemu/RA-Depth)</sup>
> - [ ] **[Dyna-DM](https://arxiv.org/abs/2206.03799), arXiv 2022.** <sup>[**`[Code]`**](https://github.com/kieran514/dyna-dm)</sup>
> - [ ] **[Lite-Mono](https://arxiv.org/abs/2211.13202), arXiv 2022.** <sup>[**`[Code]`**](https://github.com/noahzn/Lite-Mono)</sup>
> - [ ] **[TriDepth](https://arxiv.org/abs/2210.00411), WACV 2023.** <sup>[**`[Code]`**](https://github.com/xingyuuchen/tri-depth)</sup>
> - [ ] **[FreqAwareDepth](https://arxiv.org/abs/2210.05479), WACV 2023.** <sup>[**`[Code]`**](https://github.com/xingyuuchen/freq-aware-depth)</sup>

</details>

<details open>
<summary>&nbsp<b>Fully-Supervised Depth Estimation</b></summary>

> - [ ] **[AdaBins](https://arxiv.org/abs/2011.14141), CVPR 2021.** <sup>[**`[Code]`**](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/adabins)</sup>
> - [ ] **[NeWCRFs](https://arxiv.org/abs/2203.01502), CVPR 2022.** <sup>[**`[Code]`**](https://github.com/aliyun/NeWCRFs)</sup>
> - [ ] **[DepthFormer](https://arxiv.org/abs/2203.14211), arXiv 2022.** <sup>[**`[Code]`**](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/depthformer)</sup>
> - [ ] **[GLPDepth](https://arxiv.org/abs/2201.07436), arXiv 2022.** <sup>[**`[Code]`**](https://github.com/vinvino02/GLPDepth)</sup>

</details>

<details open>
<summary>&nbsp<b>Semi-Supervised Depth Estimation</b></summary>

> - [ ] **[MaskingDepth](https://arxiv.org/abs/2212.10806), arXiv 2022.** <sup>[**`[Code]`**](https://github.com/KU-CVLAB/MaskingDepth)</sup>

</details>


### :house: - Indoor Depth Estimation

<details open>
<summary>&nbsp<b>Fully-Supervised Depth Estimation</b></summary>

> - [x] **[BTS](https://arxiv.org/abs/1907.10326), arXiv 2019.** <sup>[**`[Code]`**](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/bts)</sup>
> - [x] **[AdaBins](https://arxiv.org/abs/2011.14141), CVPR 2021.** <sup>[**`[Code]`**](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/adabins)</sup>
> - [x] **[DPT](https://arxiv.org/abs/2103.13413), ICCV 2021.** <sup>[**`[Code]`**](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/dpt)</sup>
> - [x] **[SimIPU](https://arxiv.org/abs/2112.04680), AAAI 2022.** <sup>[**`[Code]`**](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/simipu)</sup>
> - [ ] **[NeWCRFs](https://arxiv.org/abs/2203.01502), CVPR 2022.** <sup>[**`[Code]`**](https://github.com/aliyun/NeWCRFs)</sup>
> - [ ] **[P3Depth](https://arxiv.org/abs/2204.02091), CVPR 2022.** <sup>[**`[Code]`**](https://github.com/SysCV/P3Depth)</sup>
> - [x] **[DepthFormer](https://arxiv.org/abs/2203.14211), arXiv 2022.** <sup>[**`[Code]`**](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/depthformer)</sup>
> - [ ] **[GLPDepth](https://arxiv.org/abs/2201.07436), arXiv 2022.** <sup>[**`[Code]`**](https://github.com/vinvino02/GLPDepth)</sup>
> - [x] **[BinsFormer](https://arxiv.org/abs/2204.00987), arXiv 2022.** <sup>[**`[Code]`**](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/binsformer)</sup>

</details>

<details open>
<summary>&nbsp<b>Semi-Supervised Depth Estimation</b></summary>

> - [ ] **[MaskingDepth](https://arxiv.org/abs/2212.10806), arXiv 2022.** <sup>[**`[Code]`**](https://github.com/KU-CVLAB/MaskingDepth)</sup>

</details>


### :robot: - RoboDepth Benchmark

<p align="center">
  <img src="docs/figs/benchmark.png" align="center" width="100%">
</p>

Kindly refer to [MODEL_ZOO.md](docs/MODEL_ZOO.md) for the detailed benchmarking results and to access the pretrained weights.


## Create Corruption Sets
You can manage to create your own "RoboDepth" corrpution sets! Follow the instructions listed in [CREATE.md](docs/CREATE.md).


## TODO List
- [x] Initial release. 🚀
- [x] Add scripts for creating common corruptions.
- [x] Add download link of KITTI-C and NYUDepth2-C.
- [x] Add competition data.
- [ ] Add evaluation scripts on corruption sets.


## Citation
If you find this work helpful, please kindly consider citing our paper:

```bibtex
@ARTICLE{kong2023robodepth,
  title={RoboDepth: Robust Out-of-Distribution Depth Estimation under Corruptions},
  author={Kong, Lingdong and Xie, Shaoyuan and Hu, Hanjiang and Cottereau, Benoit and Ng, Lai Xing and Ooi, Wei Tsang},
  journal={arXiv preprint arXiv:23xx.xxxxx}, 
  year={2023},
}
```


## License
<a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by-nc-sa/4.0/80x15.png" /></a>
<br />
This work is under the <a rel="license" href="http://creativecommons.org/licenses/by-nc-sa/4.0/">Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License</a>.

## Acknowledgements
This project is supported by [DesCartes](https://descartes.cnrsatcreate.cnrs.fr/), a [CNRS@CREATE](https://www.cnrsatcreate.cnrs.fr/) program on Intelligent Modeling for Decision-Making in Critical Urban Systems.


<p align="center">
  <img src="docs/figs/ack.png" align="center" width="100%">
</p>
