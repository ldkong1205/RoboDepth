<img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/logo2.png" align="right" width="34%">

# Validity Assessment

### Study 1: Pixel Distribution
**Goal:** Assuming that a corruption simulation is realistic enough to reflect real-world situations, the distribution of a corrupted "clean" set should be similar to that of the real-world corruption set.

**Approach:** We validate this using [ACDC](https://acdc.vision.ee.ethz.ch/news) <sup>\[R1\]</sup>, [nuScenes](https://www.nuscenes.org/nuscenes) <sup>\[R2\]</sup>, [Cityscapes](https://www.cityscapes-dataset.com/) <sup>\[R3\]</sup>, and [Foggy-Cityscapes](https://people.ee.ethz.ch/~csakarid/SFSU_synthetic/) <sup>\[R4\]</sup>, since these datasets contain:
1. real-world corruption data;
2. clean data collected by the same sensor types from the same physical locations.

We do this to ensure that there is no extra distribution shift from aspects like sensor difference (e.g. FOVs and resolutions) and location discrepancy (e.g. environmental and semantic changes).


| | | | |
| :-: | :-: | :-: | :-: | 
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_real_night.png" width="220"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_real_snow.png" width="220"> | 
| **Real Dark (ACDC-Night)** | **Real Snow (ACDC-Snow)** | **Real Dark (nuScenes-Night)** | **Real Fog (Foggy-Cityscapes)** |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-dark_1.png" width="220"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-snow_1.png" width="220"> |
| Synthetic Dark (Level 1) | Synthetic Snow (Level 1) |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-dark_2.png" width="220"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-snow_2.png" width="220"> |
| Synthetic Dark (Level 2) | Synthetic Snow (Level 2) |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-dark_3.png" width="220"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-snow_3.png" width="220"> |
| Synthetic Dark (Level 3) | Synthetic Snow (Level 3) |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-dark_4.png" width="220"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-snow_4.png" width="220"> |
| Synthetic Dark (Level 4) | Synthetic Snow (Level 4) |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-dark_5.png" width="220"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/validity/acdc_syn-snow_5.png" width="220"> |
| Synthetic Dark (Level 5) | Synthetic Snow (Level 5) |

**References:**
- \[R1\] C. Sakaridis, D. Dai, and L. V. Gool. "ACDC: The adverse conditions dataset with correspondences for semantic driving scene understanding." ICCV, 2021.
- \[R2\] C., Holger, V. Bankiti, A. H. Lang, S. Vora, V. E. Liong, Q. Xu, A. Krishnan, Y. Pan, G. Baldan, and O. Beijbom. "nuScenes: A multimodal dataset for autonomous driving." CVPR, 2020.
- \[R3\] M. Cordts, M. Omran, S. Ramos, T. Rehfeld, M. Enzweiler, R. Benenson, U. Franke, S. Roth, and B. Schiele. "The CityScapes dataset for semantic urban scene understanding." CVPR, 2016.
- \[R4\] C. Sakaridis, D. Dai, and L. V. Gool. “Semantic foggy scene understanding with synthetic data.” IJCV, 2018.

<hr>

### Study 2: Robust Fine-Tuning

