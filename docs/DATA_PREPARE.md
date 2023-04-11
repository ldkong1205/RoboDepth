<img src="../docs/figs/logo2.png" align="right" width="34%">

# Data Preparation

## Overall Structure
```shell
└── RoboDepth 
      │── kitti_data
      │    │── 2011_09_26
      │    │── ...
      │    │── kitti_c
      │    └── val
      │── cityscapes
      │    ├── camera
      │    │   ├── train
      │    │   └── val
      │    ├── disparity_trainvaltest
      │    │   └── disparity
      │    ├── leftImg8bit_trainvaltest
      │    │   └── leftImg8bit
      │    └── split_file.txt
      │── nyu
      │    │── basement_0001a
      │    │── basement_0001b
      │    │── ...
      │    │── nyu_c
      │    └── split_file.txt
      └── ...
```


## Outline
- [KITTI](#kitti)
- [KITTI-C](#kitti-c)
- [Cityscapes](#cityscapes)
- [Cityscapes-C](#cityscapes-c)
- [NYUDepth2](#nyudepth2)
- [NYUDepth2-C](#nyudepth2-c)



## KITTI
You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```
Then unzip with:
```shell
cd kitti_data/
unzip "*.zip"
cd ..
```
:dart: This dataset weighs about **175GB**, so make sure you have enough space to `unzip` too!

The `train/test/validation` splits are defined in the `splits/` folder.
By default, the code will train a depth estimation model using [Zhou's subset](https://github.com/tinghuiz/SfMLearner) of the standard Eigen split of KITTI, which is designed for monocular training.
You can also train a model using the new [benchmark split](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) or the [odometry split](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) by setting the `--split` flag.


## KITTI-C
The corrupted KITTI test sets under Eigen split can be downloaded from Google Drive with [this](https://drive.google.com/file/d/1Ohyh8CN0ZS7gc_9l4cIwX4j97rIRwADa/view?usp=sharing) link.

Alternatively, you can directly download them to the server by running:
```shell
cd kitti_data/
```
```shell
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1Ohyh8CN0ZS7gc_9l4cIwX4j97rIRwADa' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1Ohyh8CN0ZS7gc_9l4cIwX4j97rIRwADa" -O kitti_c.zip && rm -rf /tmp/cookies.txt
```
Then unzip with:
```shell
unzip kitti_c.zip
```
:dart: This dataset weighs about **12GB**, make sure you have enough space to `unzip` too!


## Cityscapes
Coming soon.


## Cityscapes-C
Coming soon.


## NYUDepth2
You can download the [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) from Google Drive with [this](https://drive.google.com/file/d/1wC-io-14RCIL4XTUrQLk6lBqU2AexLVp/view?usp=share_link) link. 

Alternatively, you can directly download it to the server by running:
```shell
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wC-io-14RCIL4XTUrQLk6lBqU2AexLVp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wC-io-14RCIL4XTUrQLk6lBqU2AexLVp" -O nyu.zip && rm -rf /tmp/cookies.txt
```
Then unzip with:
```shell
unzip nyu.zip
```
:dart: This dataset weighs about **6.2GB**, which includes 24231 image-depth pairs as the training set and the standard 654 images as the validation set.


## NYUDepth2-C
Coming soon.


