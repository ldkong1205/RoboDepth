<br />
<p align="center">
  <img src="docs/figs/logo.png" align="center" width="32%">
  
  <h3 align="center"><strong>RoboDepth: Robust Out-of-Distribution Depth Estimation under Common Corruptions</strong></h3>

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
  
  <a href="" target='_blank'>
    <img src="https://img.shields.io/badge/%E4%B8%AD%E8%AF%91%E7%89%88-%F0%9F%90%BC-red">
  </a>
</p>


## About
**RoboDepth** is a comprehensive **evaluation benchmark** designed for probing the **robustness** of **monocular** and **stereo** depth estimation algorithms. It includes **18 common corruption types**, ranging from weather and lighting conditions, sensor failure and movement, and noises during data processing.

<p align="center">
  <img src="docs/figs/taxonomy.png" align="center" width="90%">
</p>


## Updates
- [2022.11] - We are organizing the [1st RoboDepth Competition](https://robodepth.github.io/) at [ICRA 2023](https://www.icra2023.org/). Join the challenge today! :raising_hand:

## Outline
- [Installation](#installation)
- [Data Preparation](#data-preparation)
- [Getting Started](#getting-started)
- [Model Zoo](#model-zoo)
- [Create Corruption Sets](#create-corruption-sets)
- [TODO List](#todo-list)
- [Citation](#citation)
- [License](#license)
- [Acknowledgement](#acknowledgement)


## Installation
Please refer to [INSTALL.md](docs/INSTALL.md) for the installation details.


## Data Preparation
### Benchmark

Please refer to [DATA_PREPARE.md](docs/DATA_PREPARE.md) for the details to prepare the [KITTI](http://www.cvlibs.net/datasets/kitti/raw_data.php), [KITTI-C](), [Cityscapes](), and [NYUDepth2]() datasets.

### Competition @ ICRA 2023
Please refer to [COMPETITION.md](competition/COMPETITION.md) for the details to prepare the training and evaluation data 


## Getting Started
Please refer to [GET_STARTED.md](docs/GET_STARTED.md) to learn more usage about this codebase.


## Model Zoo



## Create Corruption Sets
You can manage to create your own "RoboDepth" corrpution sets! Follow the instructions listed in [CREATE.md](docs/CREATE.md).


## TODO List
- [x] Initial release. 🚀
- [x] Add scripts for creating common corruptions.
- [x] Add download link of KITTI-C.
- [x] Add competition data.
- [ ] Add evaluation scripts on corruption sets.


## Citation
If you find this work helpful, please kindly consider citing our paper:

```bibtex
@ARTICLE{kong2022robodepth,
  title={RoboDepth: Robust Out-of-Distribution Depth Estimation under Common Corruptions},
  author={Kong, Lingdong and Xie, Shaoyuan and Hu, Hanjiang and Cottereau, Benoit and Ng, Lai Xing and Ooi, Wei Tsang},
  journal={arXiv preprint arXiv:2212.xxxxx}, 
  year={2022},
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
