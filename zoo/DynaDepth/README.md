# DynaDepth

This is the official PyTorch implementation for **[Towards Scale-Aware, Robust, and Generalizable Unsupervised Monocular Depth Estimation by Integrating IMU Motion Dynamics], ECCV2022**

If you find this work useful in your research, please consider citing our paper:
```
@inproceedings{zhang2022towards,
  title={Towards scale-aware, robust, and generalizable unsupervised monocular depth estimation by integrating IMU motion dynamics},
  author={Zhang, Sen and Zhang, Jing and Tao, Dacheng},
  booktitle={European Conference on Computer Vision},
  pages={143--160},
  year={2022},
  organization={Springer}
}
```

## Method Overview
![](assets/framework.png)

## Results on KITTI
![](assets/result_1.png)
![](assets/result_2.png)

## Generalization on Make3D
![](assets/result_3.png)

## Data Preparation

1. Download both the raw (unsync) and the sync kitti datasets from https://www.cvlibs.net/datasets/kitti/raw_data.php. For each sequence, you will have two folders ```XXX_extract/``` and ```XXX_sync```, e.g. ```2011_10_03/2011_10_03_drive_0042_extract``` and ```2011_10_03/2011_10_03_drive_0042_sync```
2. The experiments are performed using the data from the sync kitti dataset (```XXX_sync/```). Since the imu (```oxt/```) in the sync dataset is sampled at the same frequency of the images, we need to perform a matching preprocessing step using the imu data in the raw dataset to get the corresponding imu data at the original frequency. 

* You can achieve this by using ```python match_kitti_imu.py```
* What you need to do: (1) Modify ```line 71-76``` to get the sequence names of your own setting (2) Modify ```line 89-90``` to your own path to the raw and the sycn datasets
* The matched results will be saved in ```matched_oxts\``` under each sequence folder ```XXX_sync```
* A 5ms drift is allowed for current matching process. You can modify ```line 153``` if you are not happy about this setting
* Note that we directly match the imu data using the timestamps, while ignoring potential time asynchronization between the imu and the camera timing systems. 

3. Since the unsync dataset is quite large to download, we also provide our preprocessed imu files in the following link: https://pan.baidu.com/s/1971KrQEHw5kVRy_Y4Lj5FA  pwd:80pz

4. For the image preprocessing, we follow the practice in https://github.com/nianticlabs/monodepth2 to convert the image format from png to jpg for a smaller image size:

```
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```

5. Since I only did preprocessing for once at the beginning of this project, please remind me by raising a new issue if I miss anything here


## Training

This codebase is developed under PyTorch-1.4.0, CUDA-10.0, and Ubuntu-18.04.1. 

You can train our full model with:

```shell
python train.py --data_path YOUR_PATH_TO_DATA --use_ekf --num_layers 50
```

To use ResNet-18 rather than ResNet-50 as the backbone, you can change ```--num_layer``` to ```18```

To disable the ekf fusion and use the IMU-related losses only, you can simply remove ```--use_ekf```

To use loss weights other than the default setting, you can manipulate with the options, e.g.,
* ```--imu_warp_weight 0.5 --imu_consistency_weight 0.01```
* ```--velo_weight 0.001 --gravity_weight 0.001```

## Evaluation 

You can evaluate on the KITTI test set with:

```shell
python evaluate_depth.py --num_layer 50 --load_weights_folder YOUR_PATH_TO_MODEL_WEIGHTS --post_process
```

By default, we report the learnt scale without the median scaling trick. Use ```--eval_mono``` if you want to test the performance with median scaling

For evaluation without post processing, simply remove ```--post_process```. 

To evaluate the models with ResNet-18 backbone, change ```--num_layer``` to ```18``` accordingly.

To evaluate the models on Make3D, use ```evaluate_make3d.py``` with the same arguments as ```evaluate_depth.py```. But you need to change the variable ```main_path``` in ```read_make3d()``` to your own path that contains test images of Make3D.

## Our pretrained models
The full pretrained models corresponding to the results in our ECCV paper can be downloaded from the following links:

DynaDepth R18: https://pan.baidu.com/s/1ksP2m-6rQ_PkBTLmjAAuLQ  pwd:xc5h

DynaDepth R50: https://pan.baidu.com/s/1X7OAOKFZ4fw3crOx6bn4ZA  pwd:c3kj


## Acknowledgment
This repo is built upon the excellent works of [monodepth2](https://github.com/nianticlabs/monodepth2), [deep_ekf_vio](https://github.com/lichunshang/deep_ekf_vio), and [liegroups](https://github.com/utiasSTARS/liegroups). The borrowed codes are licensed under their original license respectively.
