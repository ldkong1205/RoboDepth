<img src="../docs/figs/logo2.png" align="right" width="34%">

# Data Preparation

### Overall Structure
```shell
└── RoboDepth 
      │── kitti_data
      │    │── 2011_09_26
      │    │── ...
      │    │── kitti_c
      │    └── val
      │── cityscapes
      │── nyu_depth2
      └── ...
```


### KITTI
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
**Warning:** This dataset weighs about **175GB**, so make sure you have enough space to `unzip` too!

The `train/test/validation` splits are defined in the `splits/` folder.
By default, the code will train a depth estimation model using [Zhou's subset](https://github.com/tinghuiz/SfMLearner) of the standard Eigen split of KITTI, which is designed for monocular training.
You can also train a model using the new [benchmark split](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) or the [odometry split](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) by setting the `--split` flag.


### KITTI-C
The corrupted KITTI test sets under Eigen split can be downloaded from Google Drive with [this](https://drive.google.com/file/d/1NJN28mApjIa0EuRiVDyZm9VMFYp7Eqjk/view?usp=sharing) link. You can directly download them to the server by running:
```shell
cd kitti_data/
```
```shell
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1NJN28mApjIa0EuRiVDyZm9VMFYp7Eqjk' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1NJN28mApjIa0EuRiVDyZm9VMFYp7Eqjk" -O kitti_c.zip && rm -rf /tmp/cookies.txt
```
Then unzip with:
```shell
unzip kitti_c.zip
```
This dataset weighs about **12GB**, make sure you have enough space to `unzip` too!


### Cityscapes
Coming soon.


### NYUDepth2
Coming soon.
