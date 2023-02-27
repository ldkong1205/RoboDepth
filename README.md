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
- [Benchmark](#benchmark)
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


## Benchmark

### Metrics

The following metrics are consistently used in our benchmark:
- **Absolute Relative Difference (the lower the better):** $\text{Abs Rel} = \frac{1}{|D|}\sum_{pred\in D}\frac{|gt - pred|}{gt}$ .
- **Accuracy (the higher the better):** $\delta_t = \frac{1}{|D|}|\{\ pred\in D | \max{(\frac{gt}{pred}, \frac{pred}{gt})< 1.25^t}\}| \times 100\\%$ .
- **Depth Estimation Error (the lower the better):**
  - $\text{DEE}_1 = \text{Abs Rel} - \delta_1 + 1$ ;
  - $\text{DEE}_2 = \frac{\text{Abs Rel} - \delta_1 + 1}{2}$ ;
  - $\text{DEE}_3 = \frac{\text{Abs Rel}}{\delta_1}$ .

The first *Depth Estimation Error* term ($\text{DEE}_1$) is adopted as the main indicator for evaluating model performance in our RoboDepth benchmark. The following two metrics are adopted to compare between models' robustness:
- **mCE (the lower the better):** The *average corruption error* (in percentage) of a candidate model compared to the baseline model, which is calculated among all corruption types across five severity levels.
- **mRR (the higher the better):** The *average resilience rate* (in percentage) of a candidate model compared to its "clean" performance, which is calculated among all corruption types across five severity levels.

Symbol <sup>:star:</sup> denotes the baseline model adopted in *mCE* calculation.


### KITTI-C 

| Model | Modality | mCE (%) | mRR (%) | Clean | Bright | Dark | Fog | Frost | Snow | Contrast | Defocus | Glass | Motion | Zoom | Elastic| Quant| Gaussian | Impulse | Shot | ISO | Pixelate | JPEG | 
| :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| [MonoDepth2<sub>R18</sub>]()<sup>:star:</sup> | Mono | 100.00 |  |  0.238 | 0.259     | 0.561     | 0.311     | 0.553     | 1.023     | 0.373     | 0.487     | 0.484     | 0.433     | 0.402     | 0.258     | 0.386     | 0.768     | 0.779     | 0.681     | 0.776     | 0.289     | 0.391     | 
| [MonoDepth2<sub>R18+no_pt</sub>]() | Mono | | | 0.287 | 0.366     | 0.686     | 0.622     | 0.624     | 0.798     | 0.831     | 0.508     | 0.464     | 0.398     | 0.414     | 0.296     | 0.423     | 0.882     | 0.905     | 0.804     | 0.907     | 0.306     | 0.342     |
| [MonoDepth2<sub>R18+HR</sub>]() | Mono |  | | 0.229 | 0.259     | 0.751     | 0.311     | 0.543     | 1.165     | 0.428     | 0.786     | 0.514     | 0.461     | 0.463     | 0.246     | 0.431     | 0.652     | 0.703     | 0.633     | 0.688     | 0.276     | 0.395     |
| [MonoDepth2<sub>R50</sub>]() | Mono |  |  | 
| [MaskOcc]() | Mono |  |  | 0.235 | 0.259     | 0.571     | 0.308     | 0.566     | 0.984     | 0.399     | 0.637     | 0.590     | 0.455     | 0.402     | 0.257     | 0.369     | 0.805     | 0.821     | 0.729     | 0.833     | 0.285     | 0.353     |
| [DNet]() | Mono |  |  | 0.237 | 0.256     | 0.528     | 0.313     | 0.634     | 1.007     | 0.419     | 0.696     | 0.640     | 0.485     | 0.429     | 0.262     | 0.377     | 0.724     | 0.732     | 0.651     | 0.715     | 0.290     | 0.381     |
| [CADepth]() | Mono | | | 0.217 | 0.241     | 0.599     | 0.283     | 0.649     | 1.058     | 0.386     | 0.712     | 0.695     | 0.570     | 0.416     | 0.242     | 0.384     | 0.847     | 0.866     | 0.766     | 0.896     | 0.289     | 0.390     |
| [HR-Depth]() | Mono |  |  | 0.225 | 0.242     | 0.578     | 0.302     | 0.558     | 0.961     | 0.425     | 0.711     | 0.600     | 0.526     | 0.448     | 0.247     | 0.375     | 0.726     | 0.745     | 0.671     | 0.749     | 0.269     | 0.351     |
| [DIFFNet]() | Mono |  |  | 0.205 | 0.223     | 0.443     | 0.262     | 0.399     | 0.705     | 0.322     | 1.025     | 0.659     | 0.560     | 0.393     | 0.229     | 0.331     | 0.583     | 0.532     | 0.510     | 0.540     | 0.270     | 0.404     |
| [ManyDepth]() | Mono |  |  | 0.247 | 0.270     | 0.548     | 0.337     | 0.575     | 0.959     | 0.455     | 0.508     | 0.558     | 0.422     | 0.388     | 0.267     | 0.378     | 0.859     | 0.900     | 0.774     | 0.904     | 0.295     | 0.365     |
| [FSRE-Depth]() | Mono |  |  | 0.219 | 0.255     | 0.523     | 0.277     | 0.474     | 0.786     | 0.339     | 0.583     | 0.546     | 0.428     | 0.369     | 0.239     | 0.357     | 0.799     | 0.828     | 0.740     | 0.814     | 0.294     | 0.447     |
| |
| [MonoDepth2<sub>R18</sub>]() | Stereo | | | 0.246 | 0.266     | 0.696     | 0.322     | 0.609     | 1.031     | 0.468     | 0.780     | 0.663     | 0.528     | 0.418     | 0.270     | 0.399     | 0.985     | 1.018     | 0.927     | 0.985     | 0.288     | 0.387     |
| [MonoDepth2<sub>R18+no_pt</sub>]() | Stereo | | | 0.299 | 0.362     | 0.843     | 0.583     | 0.704     | 0.871     | 0.685     | 0.532     | 0.464     | 0.434     | 0.459     | 0.312     | 0.472     | 1.078     | 1.128     | 1.042     | 1.112     | 0.328     | 0.356     |
| [MonoDepth2<sub>R18+HR</sub>]() | Stereo | | | 0.234 | 0.265     | 0.569     | 0.334     | 0.712     | 1.059     | 0.476     | 0.865     | 0.623     | 0.558     | 0.492     | 0.260     | 0.412     | 0.686     | 0.686     | 0.644     | 0.687     | 0.299     | 0.418     |
| [DepthHints]() | Stereo |  |  |  0.226 | 0.247     | 0.621     | 0.275     | 0.642     | 1.030     | 0.328     | 0.700     | 0.820     | 0.526     | 0.392     | 0.259     | 0.383     | 0.880     | 0.893     | 0.823     | 0.910     | 0.315     | 0.384     |
| [DepthHints<sub>HR</sub>]() | Stereo |  |  | 0.208 | 0.243     | 0.564     | 0.281     | 0.633     | 0.960     | 0.359     | 0.918     | 0.725     | 0.640     | 0.523     | 0.236     | 0.365     | 0.793     | 0.842     | 0.760     | 0.849     | 0.282     | 0.366     |
| [DepthHints<sub>HR+no_pt</sub>]() | Stereo |  |  | 0.267 | 0.346     | 0.951     | 0.603     | 0.747     | 0.926     | 0.786     | 0.714     | 0.577     | 0.482     | 0.462     | 0.285     | 0.495     | 1.225     | 1.317     | 1.198     | 1.385     | 0.303     | 0.381     |
| |
| [MonoDepth2<sub>R18</sub>]() | M+S | | |  0.232 | 0.255     | 0.808     | 0.301     | 0.590     | 1.073     | 0.398     | 0.895     | 0.692     | 0.566     | 0.408     | 0.256     | 0.407     | 1.154     | 1.210     | 1.121     | 1.259     | 0.271     | 0.358     |
| [MonoDepth2<sub>R18+no_pt</sub>]() | M+S | | | 0.291 | 0.386     | 0.920     | 0.656     | 0.842     | 0.857     | 0.881     | 0.457     | 0.442     | 0.431     | 0.459     | 0.306     | 0.457     | 1.140     | 1.193     | 1.098     | 1.213     | 0.322     | 0.354     |
| [MonoDepth2<sub>R18+HR</sub>]() | M+S | | | 0.229 | 0.259     | 0.751     | 0.311     | 0.543     | 1.165     | 0.428     | 0.786     | 0.514     | 0.461     | 0.463     | 0.246     | 0.431     | 0.652     | 0.703     | 0.633     | 0.688     | 0.276     | 0.395     |
| [CADepth]() | M+S |  |  | 0.221 | 0.247     | 0.715     | 0.275     | 0.623     | 1.111     | 0.337     | 0.676     | 0.825     | 0.520     | 0.385     | 0.252     | 0.372     | 1.093     | 1.119     | 1.047     | 1.165     | 0.290     | 0.384     |


### NYUDepth2-C

| Model | mCE (%) | mRR (%) | Clean | Bright | Dark | Fog | Frost | Snow | Contrast | Defocus | Glass | Motion | Zoom | Elastic| Quant| Gaussian | Impulse | Shot | ISO | Pixelate | JPEG | 
| :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| [BTS<sub>R50</sub>]() |
| [AdaBins<sub>R50</sub>]()<sup>:star:</sup> | 
| [AdaBins<sub>EfficientB5</sub>]() |
| [DPT<sub>ViT-B</sub>]() |
| [SimIPU<sub>R50+no_pt</sub>]() |
| [SimIPU<sub>R50+imagenet</sub>]() |
| [SimIPU<sub>R50+kitti</sub>]() |
| [SimIPU<sub>R50+waymo</sub>]() |
| [DepthFormer<sub>SwinT_w7_1k</sub>]() | 
| [DepthFormer<sub>SwinT_w7_22k</sub>]() | 


<p align="center">
  <img src="docs/figs/benchmark.png" align="center" width="100%">
</p>

For more detailed benchmarking results and to access the pretrained weights used in robustness evaluation, kindly refer to [MODEL_ZOO.md](docs/MODEL_ZOO.md).


## Create Corruption Sets
You can manage to create your own "RoboDepth" corrpution sets! Follow the instructions listed in [CREATE.md](docs/CREATE.md).


## TODO List
- [x] Initial release. 🚀
- [x] Add scripts for creating common corruptions.
- [x] Add download link of KITTI-C and NYUDepth2-C.
- [x] Add competition data.
- [x] Add benchmarking results.
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
