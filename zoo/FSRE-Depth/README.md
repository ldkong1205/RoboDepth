# FSRE-Depth
This is a Python3 / PyTorch implementation of FSRE-Depth, as described in the following paper:

> **Fine-grained Semantics-aware Representation Enhancement for Self-supervisedMonocular Depth Estimation**
>![overview](https://user-images.githubusercontent.com/30494126/136926985-af8c3651-4503-402b-9677-f623f8b0fd95.PNG)
> *Hyunyoung Jung, Eunhyeok Park and Sungjoo Yoo*
>
> ICCV 2021 (oral)
> 
> [arXiv pdf](http://arxiv.org/abs/2108.08829)


The code was implemented based on [Monodepth2](https://github.com/nianticlabs/monodepth2).

## Setup
This code was implemented under torch==1.3.0 and torchvision==0.4.1, using two NVIDIA TITAN Xp gpus with distrutibted training. Different version may produce different results.
```
pip install -r requirements.txt
```
## Dataset
[KITTI Raw Data](http://www.cvlibs.net/datasets/kitti/raw_data.php) and pre-computed [segmentation images](https://drive.google.com/file/d/1FNxJzGTfP1O_pUX9Va7d0dqZWtRi833X/view?usp=sharing) are required for training. 

```
KITTI/
    ├── 2011_09_26/             
    ├── 2011_09_28/                    
    ├── 2011_09_29/
    ├── 2011_09_30/
    ├── 2011_10_03/
    └── segmentation/   # download and unzip "segmentation.zip" 
```

## Training
For training the full model, run the command as below:
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node 2 --master_port YOUR_PORT_NUMBER train_ddp.py --data_path YOUR_KITTI_DATA_PATH
```

## Evaluation
The ground truth depth maps should be prepared prior to evaluation. 
```
python export_gt_depth.py --data_path YOUR_KITTI_DATA_PATH --split eigen
```

MODEL_DIR should be configured as below:

```
MODEL_DIR
    ├── encoder.pth  # required      
    ├── decoder.pth  # required             
    ├── ...
```

Run the evaluation command.
```
python evaluate_depth.py --load_weights_folder MODEL_DIR --data_path YOUR_KITTI_DATA_PATH
```

## Download Models

| Backbone | Input  |Download                                                                                              |AbsRel | SqRel | Rms | RmsLog | delta < 1.25 |    delta < 1.25^2 |   delta < 1.25^3  |
|----------|-------------|--------------------------------------------------------------------------------------------------|--------|--------|--------|--------|--------|--------|--------|
| ResNet-18| 192 x 640   |[Drive (.zip)](https://drive.google.com/file/d/14uT9DyCU0UKynfBnzymaStRiL0sJmqt2/view?usp=sharing)| 0.105  |  0.708  |  4.546  |  0.182  |  0.886  |  0.964  |  0.983|         

## Reference
Please use the following citation when referencing our work:
```
@InProceedings{Jung_2021_ICCV,
    author    = {Jung, Hyunyoung and Park, Eunhyeok and Yoo, Sungjoo},
    title     = {Fine-Grained Semantics-Aware Representation Enhancement for Self-Supervised Monocular Depth Estimation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {12642-12652}
}
```
