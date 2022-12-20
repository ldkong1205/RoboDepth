# DIFFNet

This repo is for **[Self-Supervised Monocular Depth Estimation with Internal Feature Fusion(arXiv)](https://arxiv.org/pdf/2110.09482.pdf), BMVC2021**

 A new backbone for self-supervised depth estimation.

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/self-supervised-monocular-depthestimation/monocular-depth-estimation-on-kitti-eigen-1)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen-1?p=self-supervised-monocular-depthestimation)


If you think it is a useful work, please consider citing it.
```
@inproceedings{zhou_diffnet,
    title={Self-Supervised Monocular Depth Estimation with Internal Feature Fusion},
    author={Zhou, Hang and Greenwood, David and Taylor, Sarah},
    booktitle={British Machine Vision Conference (BMVC)},
    year={2021}
    }

```
## Update:
- [16-05-2022] Adding cityscapes trainining and testing based on [Manydepth](https://github.com/nianticlabs/manydepth). 

- [22-01-2022] A model diffnet_649x192 uploaded (slightly improved than that of orginal paper)
- [07-12-2021] A multi-gpu training version availible on multi-gpu branch.


## Comparing with others
![](images/table1.png)

## Evaluation on selected hard cases:
![](images/table2.png)

## Trained weights on KITTI
* Please Note: the results of diffnet_1024x320_ms are not reported in paper *

| Methods |abs rel|sq rel| RMSE |rmse log | D1 | D2 | D3 |
| :----------- | :-----: | :----: | :---: | :------: | :--------: |:--------: |:--------: |
 [1024x320](https://drive.google.com/file/d/1SuyBMS3ZLYuZwgyGSpmNrag7ESjRUC52/view?usp=sharing)|0.097|0.722|4.345|0.174|0.907|0.967|0.984|
 [1024_320_ms](https://drive.google.com/file/d/1VR0BYXKyclvv1Gq2XcQCR-fvJuFQ80SI/view?usp=sharing)|0.094|0.678|4.250|0.172|0.911|0.968|0.984|
 [1024x320_ms_ttr](https://drive.google.com/file/d/1u4pizvk9xZ8bbyWLyjd0m_9hnm_mO9-Q/view?usp=sharing)|0.079|0.640|3.934|0.159|0.932|0.971|0.984 | 
 [640x192](https://drive.google.com/file/d/1ZQPZWsIy_KyjV-Et6FSCOPM4iATjDPn-/view?usp=sharing)|0.102|0.753|4.459|0.179|0.897|0.965|0.983|
 [640x192_ms](https://drive.google.com/file/d/1_vh1F_cabTlEjBGXkHZOpAB1CMLmosxg/view?usp=sharing)|0.101|0.749|4.445|0.179|0.898|0.965|0.983|

## Setting up before training and testing

- Data preparation: please refer to [monodepth2](https://github.com/nianticlabs/monodepth2)

## Training:

```
sh start2train.sh
```

## Testing:

```
sh disp_evaluation.sh
```
## Infer a single depth map from a RGB:

```
sh test_sample.sh
```


#### Acknowledgement
 Thanks the authors for their works:
 - [monodepth2](https://github.com/nianticlabs/monodepth2)
 - [HRNet](https://github.com/HRNet/HRNet-Semantic-Segmentation)

