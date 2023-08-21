<img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/logo2.png" align="right" width="34%">

# Validity Assessment
### Outline
- [Study 1: Pixel Distribution](#study-1-pixel-distribution)
- [Study 2: Robust Fine-Tuning](#study-2-robust-fine-tuning)


## Study 1: Pixel Distribution
**Goal:** Assuming that a corruption simulation is realistic enough to reflect real-world situations, the distribution of a corrupted "clean" set should be similar to that of the real-world corruption set.

**Approach:** We validate this using [ACDC](https://acdc.vision.ee.ethz.ch/news) <sup>\[R1\]</sup>, [nuScenes](https://www.nuscenes.org/nuscenes) <sup>\[R2\]</sup>, [Cityscapes](https://www.cityscapes-dataset.com/) <sup>\[R3\]</sup>, and [Foggy-Cityscapes](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/) <sup>\[R4\]</sup>, since these datasets contain:
1. real-world corruption data;
2. clean data collected by the same sensor types from the same physical locations.

We simulate corruptions using "clean" images and compare the distribution patterns with their corresponding real-world corrupted data. We do this to ensure that there is no extra distribution shift from aspects like sensor difference (e.g. FOVs and resolutions) and location discrepancy (e.g. environmental and semantic changes).

| **Real Dark (ACDC-Night)** | **Real Snow (ACDC-Snow)** | **Real Dark (nuScenes-Night)** | **Real Fog (Foggy-Cityscapes)** |
| :-: | :-: | :-: | :-: | 
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_real_night.png" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_real_snow.png" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/nusc_real_night.png" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/cityscapes_real_fog.png" width="225"> | 
| Synthetic Dark (Level 1) | Synthetic Snow (Level 1) | Synthetic Dark (Level 1) | Synthetic Fog (Level 1) |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-dark_1.png" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-snow_1.png" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/nusc_syn-dark_1.jpg" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/cityscapes_fog_1.png" width="225"> |
| Synthetic Dark (Level 2) | Synthetic Snow (Level 2) | Synthetic Dark (Level 2) | Synthetic Fog (Level 2) |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-dark_2.png" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-snow_2.png" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/nusc_syn-dark_2.png" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/cityscapes_fog_2.png" width="225"> |
| Synthetic Dark (Level 3) | Synthetic Snow (Level 3) | Synthetic Dark (Level 3) | Synthetic Fog (Level 3) |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-dark_3.png" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-snow_3.png" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/nusc_syn-dark_3.png" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/cityscapes_fog_3.png" width="225"> |
| Synthetic Dark (Level 4) | Synthetic Snow (Level 4) | Synthetic Dark (Level 4) | Synthetic Fog (Level 4) |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-dark_4.png" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-snow_4.png" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/nusc_syn-dark_4.png" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/cityscapes_fog_4.png" width="225"> |
| Synthetic Dark (Level 5) | Synthetic Snow (Level 5) | Synthetic Dark (Level 5) | Synthetic Fog (Level 5) |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-dark_5.png" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-snow_5.png" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/nusc_syn-dark_5.jpg" width="225"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/cityscapes_fog_5.png" width="225"> |

**References:**
- \[R1\] C. Sakaridis, D. Dai, and L. V. Gool. "ACDC: The adverse conditions dataset with correspondences for semantic driving scene understanding." ICCV, 2021.
- \[R2\] C., Holger, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom. "nuScenes: A multimodal dataset for autonomous driving." CVPR, 2020.
- \[R3\] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. "The CityScapes dataset for semantic urban scene understanding." CVPR, 2016.
- \[R4\] C. Sakaridis, D. Dai, and L. V. Gool. “Semantic foggy scene understanding with synthetic data.” IJCV, 2018.


## Study 2: Robust Fine-Tuning

**Goal:** Assuming that a corruption simulation is realistic enough to reflect real-world situations, a corruption-augmented model should achieve better generalizability than the "clean" model when tested on real-world corruption datasets.

**Approach:** We validate this using [nuScenes](https://www.nuscenes.org/nuscenes), [nuScenes-Night](https://www.nuscenes.org/nuscenes), and [Foggy-Cityscapes](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/). We adopt [MonoDepth2](https://github.com/nianticlabs/monodepth2) as the baseline, which is trained on [KITTI](https://www.cvlibs.net/datasets/kitti/raw_data.php) and fine-tuned with corruptions with a small learning rate. We also test training with corruptions from scratch and find the performance is similar to fine-tuning.

### nuScenes
| Train | Backbone | Resolution | CorruptAug | Abs Rel | Sq Rel | RMSE | RMSE log | a1 | a2 | a3 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| KITTI | ResNet-18 | 640x192 | No  | 0.304 | 3.472 | 9.068 | 0.409 | 0.563 | 0.794 | 0.890 |
| KITTI | ResNet-18 | 640x192 | Yes | 0.297 | 2.991 | 8.790 | 0.405 | 0.558 | 0.794 | 0.893 |
| KITTI | ResNet-50 | 640x192 | No  | 0.302 | 3.219 | 9.054 | 0.416 | 0.555 | 0.786 | 0.886 |
| KITTI | ResNet-50 | 640x192 | Yes | 0.294 | 2.947 | 8.754 | 0.404 | 0.565 | 0.795 | 0.892 |

### nuScenes-Night
| Train | Backbone | Resolution | CorruptAug | Abs Rel | Sq Rel | RMSE | RMSE log | a1 | a2 | a3 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| KITTI | ResNet-18 | 640x192 | No  | 0.397 | 3.408 | 8.700 | 0.513 | 0.387 | 0.659 | 0.822 |
| KITTI | ResNet-18 | 640x192 | Yes | 0.362 | 3.149 | 8.391 | 0.477 | 0.434 | 0.714 | 0.852 |
| KITTI | ResNet-50 | 640x192 | No  | 0.418 | 3.599 | 8.928 | 0.539 | 0.363 | 0.626 | 0.802 |
| KITTI | ResNet-50 | 640x192 | Yes | 0.357 | 3.128 | 8.168 | 0.462 | 0.444 | 0.723 | 0.861 |

### Foggy-Cityscapes
| Train | Backbone | Resolution | CorruptAug | Abs Rel | Sq Rel | RMSE | RMSE log | a1 | a2 | a3 |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| KITTI | ResNet-18 | 416x128 | No  | 0.421 | 7.057 | 15.207 | 0.527 | 0.360 | 0.636 | 0.806 |
| KITTI | ResNet-18 | 416x128 | Yes | 0.385 | 6.310 | 14.654 | 0.489 | 0.399 | 0.682 | 0.836 |
| KITTI | ResNet-18 | 512x256 | No  | 0.364 | 6.371 | 14.690 | 0.483 | 0.440 | 0.703 | 0.838 |
| KITTI | ResNet-18 | 512x256 | Yes | 0.349 | 5.645 | 14.723 | 0.488 | 0.434 | 0.698 | 0.834 |




