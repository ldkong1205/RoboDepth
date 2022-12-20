# Learning Monocular Depth in Dynamic Scenes via Instance-Aware Projection Consistency


[ [Install](#install) | [Datasets](#datasets) | [Training](#training) | [Models](#models) | [Evaluation](#evaluation) | [Demo](#demo) | [References](#references) | [License](#license) ]


This is the official PyTorch implementation for the system proposed in the paper :

 >**Learning Monocular Depth in Dynamic Scenes via Instance-Aware Projection Consistency**
 >
 >[**Seokju Lee**](https://sites.google.com/site/seokjucv/), [Sunghoon Im](https://sunghoonim.github.io/), [Stephen Lin](https://www.microsoft.com/en-us/research/people/stevelin/), and [In So Kweon](http://rcv.kaist.ac.kr/index.php?mid=rcv_faculty)
 >
 >**AAAI-21** [[PDF](https://arxiv.org/abs/2102.02629)] [[Project](https://sites.google.com/site/seokjucv/home/instadm)]


<p align="center">
  <img src="./misc/demo_sample.gif"/>
</p>

<p align="center">
  &Longrightarrow; <strong><em>Unified Visual Odometry</em></strong> : Our holistic visualization of depth and motion estimation from self-supervised monocular training.
</p>


### If you find our work useful in your research, please consider citing our paper :
 
```bibtex
@inproceedings{lee2021learning,
  title={Learning Monocular Depth in Dynamic Scenes via Instance-Aware Projection Consistency},
  author={Lee, Seokju and Im, Sunghoon and Lin, Stephen and Kweon, In So},
  booktitle={Proceedings of the AAAI Conference on Artificial Intelligence (AAAI)},
  year={2021}
}
```


## Install

Our code is tested with CUDA 10.2/11.0, Python 3.7.x (conda environment), and PyTorch 1.4.0/1.7.0.

At least 2 GPUs (each 12 GB) are required to train the models with `batch_size=4` and `maximum_number_of_instances_per_frame=3`.

Create a conda environment with PyTorch library as :

```bash
conda create -n my_env python=3.7.4 pytorch=1.7.0 torchvision torchaudio cudatoolkit=11.0 -c pytorch
conda activate my_env
```

Install prerequisite packages listed in :

```bash
pip3 install -r requirements.txt
```

or install manually the following packages :

```
opencv-python
imageio
matplotlib
scipy==1.1.0
scikit-image
argparse
tensorboardX
blessings
progressbar2
path
tqdm
pypng
open3d==0.8.0.0
```

Please install `torch-scatter` and `torch-sparse` following [this link](https://github.com/rusty1s/pytorch_sparse).

```bash
pip3 install torch-scatter torch-sparse -f https://pytorch-geometric.com/whl/torch-1.7.0+cu110.html
```


## Datasets

We provide our KITTI-VIS and Cityscapes-VIS dataset ([download link](https://drive.google.com/drive/folders/1tgQKHDj3tf97LZoQqXxcOF4LVHoIKXcs?usp=sharing)), which is composed of pre-processed images, auto-annotated instance segmentation, and optical flow.

- Images are pre-processed with [SC-SfMLearner](https://github.com/JiawangBian/SC-SfMLearner-Release/blob/master/scripts/run_prepare_data.sh).

- Instance segmentation is pre-processed with [PANet](https://github.com/ShuLiu1993/PANet).

- Optical flow is pre-processed with [PWC-Net](https://github.com/NVlabs/PWC-Net/tree/master/Caffe).

We associate them to operate video instance segmentation as implemented in `datasets/sequence_folders.py`.

Please allocate the dataset as the following file structure :
```
kitti_256 (or cityscapes_256)
    └ image
        └ $SCENE_DIR
    └ segmentation
        └ $SCENE_DIR
    └ flow_f
        └ $SCENE_DIR
    └ flow_b
        └ $SCENE_DIR
    ├ train.txt
    └ val.txt
```
Training and validation scenes can be randomly generated in `train.txt` and `val.txt`.


## Training

You can train the models on KITTI-VIS by running :

```bash
sh scripts/train_resnet_256_kt.sh
```

You can train the models on Cityscapes-VIS by running :

```bash
sh scripts/train_resnet_256_cs.sh
```

Please indicate the location of the dataset with `$TRAIN_SET`.

The hyperparameters (batch size, learning rate, loss weight, etc.) are defined in each script file and [default arguments](train.py) in `train.py`. Please also check our [main paper](https://arxiv.org/abs/2102.02629).

During training, checkpoints will be saved in `checkpoints/`.

You can also start a `tensorboard` session by running :
```bash
tensorboard --logdir=checkpoints/ --port 8080 --bind_all
```
and visualize the training progress by opening [https://localhost:8080](https://localhost:8080) on your browser. 

For convenience, we provide two breakpoints (supported with pdb), commented as `BREAKPOINT` in `train.py`.
Each breakpoint represents an important point in projecting the object.
<pre>
<b>BREAKPOINT-1</b> : Breakpoint after the 1st projection with camera motion. Visualize ego-warped images.
<b>BREAKPOINT-2</b> : Breakpoint after the 2nd projection with each object motion. Visualize fully-warped images and motion fields.
</pre>
You can visualize the intermediate outputs with the commented code. This will improve your visibility on debugging the code.


## Models

We provide KITTI-VIS and Cityscapes-VIS pretrained models ([download link](https://drive.google.com/drive/folders/1KLUF4MTkb85GWu8s5y0WVTKwGiE8gxC0?usp=sharing)).

The architectures are based on the ResNet18 encoder. Please see the details of them in `models/`.


Models trained under three different conditions are released :
<pre>
<b>KITTI</b> : Trained on KITTI-VIS using ImageNet (ResNet18) pretrained model.
<b>CS</b> : Trained on Cityscapes-VIS using ImageNet (ResNet18) pretrained model. This model is only for the pretraining and demo.
<b>CS+KITTI</b> : Pretrained on Cityscapes-VIS, and finetuned on KITTI-VIS.
</pre>


## Evaluation

We evaluate our depth estimation following the [KITTI Eigen split](https://arxiv.org/abs/1406.2283).
For the evaluation, it is required to download the [KITTI raw dataset](http://www.cvlibs.net/download.php?file=raw_data_downloader.zip) provided on the official website.
Tested scenes are listed in `kitti_eval/test_files_eigen.txt`.

You can evaluate the models by running :

```bash
sh scripts/run_eigen_test.sh
```

Please indicate the location of the raw dataset with `$DATA_ROOT`, and the models with `$DISP_NET`.

We demonstrate our results as follows :

| Models                                           | Abs Rel | Sq Rel | RMSE  | RMSE log | Acc 1 | Acc 2 | Acc 3 |
| :----------------------------------------------- | :-----: | :----: | :---: | :------: | :---: | :---: | :---: |
| ResNet18, 832x256, ImageNet &rightarrow; KITTI   | 0.112   | 0.777  | 4.772 | 0.191    | 0.872 | 0.959 | 0.982 |
| ResNet18, 832x256, Cityscapes &rightarrow; KITTI | 0.109   | 0.740  | 4.547 | 0.184    | 0.883 | 0.962 | 0.983 |

For convenience, we also provide precomputed depth maps in [this link](https://drive.google.com/drive/folders/1HfG9ZplSPPy42OIQAMfEeNcxovscRLOa?usp=sharing).



## Demo

We demonstrate *Unified Visual Odometry*, which shows the results of depth, ego-motion, and object motion holistically.

You can visualize them by running :

```bash
sh scripts/run_demo.sh
```

Please indicate the location of the image samples with `$SCENE`. We recommend to visualize Cityscapes scenes since it contains more dynamic objects than KITTI.

More results are demonstrated in [this link](https://youtu.be/_S4GnK8QTF4).


## References
 
* [SC-SfMLearner](https://github.com/JiawangBian/SC-SfMLearner-Release) (NeurIPS 2019, our baseline framework)

* [PANet](https://github.com/ShuLiu1993/PANet) (CVPR 2018, instance segmentation for data pre-processing)
 
* [PWC-Net](https://github.com/NVlabs/PWC-Net/tree/master/Caffe) (CVPR 2018, optical flow for data pre-processing)
 
* [PyTorch-Sparse](https://github.com/rusty1s/pytorch_sparse) (PyTorch library for sparse tensor representation)
 
* [Struct2Depth](https://github.com/tensorflow/models/blob/archive/research/struct2depth) (AAAI 2019, object scale loss)

* [Depth from Video in the Wild](https://github.com/google-research/google-research/tree/master/depth_from_video_in_the_wild) (ICCV 2019, motion field representation)


## License

The source code is released under the [MIT license](LICENSE).
 
