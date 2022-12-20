# DNet

This is the reference PyTorch implementation for training and testing depth estimation models using the method described in

> **Toward Hierarchical Self-Supervised Monocular Absolute Depth Estimation for Autonomous Driving Applications**
>
> Feng Xue, Guirong Zhuo<sup>\*</sup>, Ziyuan Huang, Wufei Fu, Zhuoyue Wu and Marcelo H. Ang Jr
>
> [IROS 2020](https://ieeexplore.ieee.org/document/9340802)

<p align="center">
  <img src="assets/demo.gif" alt="example input output gif" width="800" />
</p>


If you find our work useful in your research please consider citing our paper:

```
@inproceedings{xue2020toward,
  title={Toward hierarchical self-supervised monocular absolute depth estimation for autonomous driving applications},
  author={Xue, Feng and Zhuo, Guirong and Huang, Ziyuan and Fu, Wufei and Wu, Zhuoyue and Ang, Marcelo H},
  booktitle={2020 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={2330--2337},
  year={2020},
  organization={IEEE}
}
```

This repository is maintained by [Feng Xue](https://github.com/josephxue) and [Wufei Fu](https://github.com/fwf-lernen)


## License

The code is derived from [Monodepth v2](https://github.com/nianticlabs/monodepth2)

Copyright © Niantic, Inc. 2019. Patent Pending. All rights reserved.  
This code is for non-commercial use; please see the [license file](LICENSE) for terms.


## Setup

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
pip install tensorboardX==1.4
conda install opencv=3.3.1   # just needed for evaluation
```
We ran our experiments with PyTorch 1.2.0, CUDA 10.2, Python 3.5 and Ubuntu 16.04.


## KITTI training data

You can download the entire [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```
Then unzip with
```shell
cd kitti_data
unzip "*.zip"
cd ..
```
**Warning:** it weighs about **175GB**, so make sure you have enough space to unzip too!

Our default settings expect that you have converted the png images to jpeg with this command, **which also deletes the raw KITTI `.png` files**:
```shell
find kitti_data/ -name '*.png' | parallel 'convert -quality 92 -sampling-factor 2x2,1x1,1x1 {.}.png {.}.jpg && rm {}'
```
**or** you can skip this conversion step and train from raw png files by adding the flag `--png` when training, at the expense of slower load times.

You can also place the KITTI dataset wherever you like and point towards it with the `--data_path` flag during training and evaluation.

**Splits**

The train/test/validation splits are defined in the `splits/` folder.
By default, the code will train a depth model using [Zhou's subset](https://github.com/tinghuiz/SfMLearner) of the standard Eigen split of KITTI, which is designed for monocular training.
You can also train a model using the new [benchmark split](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) or the [odometry split](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) by setting the `--split` flag.

**Custom dataset**

You can train on a custom monocular or stereo dataset by writing a new dataloader class which inherits from `MonoDataset` – see the `KITTIDataset` class in `datasets/kitti_dataset.py` for an example.


## Training

By default models and tensorboard event files are saved to `~/tmp/<model_name>`.
This can be changed with the `--log_dir` flag.

**Monocular training:**
```shell
python train.py --model_name mono_model
```

**Stereo training:**

Our code defaults to using Zhou's subsampled Eigen training data. For stereo-only training we have to specify that we want to use the full Eigen training set – see paper for details.
```shell
python train.py --model_name stereo_model \
  --frame_ids 0 --use_stereo --split eigen_full
```

**Monocular + stereo training:**
```shell
python train.py --model_name mono+stereo_model \
  --frame_ids 0 -1 1 --use_stereo
```

### GPUs

The code can only be run on a single GPU.
You can specify which GPU to use with the `CUDA_VISIBLE_DEVICES` environment variable:
```shell
CUDA_VISIBLE_DEVICES=2 python train.py --model_name mono_model
```

### Finetuning a pretrained model

Add the following to the training command to load an existing model for finetuning:
```shell
python train.py --model_name finetuned_mono --load_weights_folder ~/tmp/mono_model/models/weights_19
```

### Other training options

Run `python train.py -h` (or look at `options.py`) to see the range of other training options, such as learning rates and ablation settings.


## Reproduction

We provide [pretrained model](https://drive.google.com/file/d/1DqDrZNt3q_U5sK5bFInzj835x-Q5S6Hh/view?usp=sharing) and [precomputed results](https://drive.google.com/file/d/1i7TXiG7p2z4blsb4sfAMdNdHIASC_NWk/view?usp=sharing) to reproduce the data mentioned in paper.  
This model achieves following results on KITTI dataset Eigen split:

| Abs Rel | Sq Rel | RMSE | RMSE log | Acc.1 | Acc.2 | Acc.3 |
|:-: | :-: | :-:| :-: | :-: | :-: | :-: |
| 0.113 | 0.864 | 4.812 | 0.191 | 0.877 | 0.960 | 0.981 |


## KITTI evaluation

To prepare the ground truth depth maps run:
```shell
python export_gt_depth.py --data_path kitti_data --split eigen
python export_gt_depth.py --data_path kitti_data --split eigen_benchmark
```
...assuming that you have placed the KITTI dataset in the default location of `./kitti_data/`.

The following example command evaluates the epoch 19 weights of a model named `mono_model`:
```shell
python evaluate_depth.py --load_weights_folder ~/tmp/mono_model/models/weights_19/ --eval_mono
```

The above command is the default evaluation mode, which evaluates depth from all the lidar ground truth and uses ground truth for scale recovery.  

If you want to recover scale from our proposed dense geometrical constrain, run the following command:
```shell
python evaluate_depth.py --load_weights_folder ~/tmp/mono_model/models/weights_19/ --eval_mono --scaling dgc
```

If you only want to evaluate depth on objects, first download our preprocessed [object mask](https://drive.google.com/drive/folders/1f8T_hZRAhmXwE6_rlX05aQUrM69aJmfu?usp=sharing) to DNet folder, and run the following command:
```shell
python evaluate_depth.py --load_weights_folder ~/tmp/mono_model/models/weights_19/ --eval_mono --eval_object
```

For stereo models, you must use the `--eval_stereo` flag (see note below):
```shell
python evaluate_depth.py --load_weights_folder ~/tmp/stereo_model/models/weights_19/ --eval_stereo
```
If you train your own model with our code you are likely to see slight differences to the publication results due to randomization in the weights initialization and data loading.

An additional parameter `--eval_split` can be set.
The three different values possible for `eval_split` are explained here:

| `--eval_split`        | Test set size | For models trained with... | Description  |
|-----------------------|---------------|----------------------------|--------------|
| **`eigen`**           | 697           | `--split eigen_zhou` (default) or `--split eigen_full` | The standard Eigen test files |
| **`eigen_benchmark`** | 652           | `--split eigen_zhou` (default) or `--split eigen_full`  | Evaluate with the improved ground truth from the [new KITTI depth benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) |
| **`benchmark`**       | 500           | `--split benchmark`        | The [new KITTI depth benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) test files. |

Because no ground truth is available for the new KITTI depth benchmark, no scores will be reported  when `--eval_split benchmark` is set.
Instead, a set of `.png` images will be saved to disk ready for upload to the evaluation server.

**External disparities evaluation**

Finally you can also use `evaluate_depth.py` to evaluate raw disparities (or inverse depth) from other methods by using the `--ext_disp_to_eval` flag:

```shell
python evaluate_depth.py --ext_disp_to_eval ~/other_method_disp.npy
```


**Note on stereo evaluation**

Our stereo models are trained with an effective baseline of `0.1` units, while the actual KITTI stereo rig has a baseline of `0.54m`. This means a scaling of `5.4` must be applied for evaluation.
In addition, for models trained with stereo supervision we disable median scaling.
Setting the `--eval_stereo` flag when evaluating will automatically disable median scaling and scale predicted depths by `5.4`.

**Odometry evaluation**

We include code for evaluating poses predicted by models trained with `--split odom --dataset kitti_odom --data_path /path/to/kitti/odometry/dataset`.

For this evaluation, the [KITTI odometry dataset](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) **(color, 65GB)** and **ground truth poses** zip files must be downloaded.
As above, we assume that the pngs have been converted to jpgs.

If this data has been unzipped to folder `kitti_odom`, a model can be evaluated with:
```shell
python evaluate_pose.py --eval_split odom_9 --load_weights_folder ./odom_split.M/models/weights_29 --data_path kitti_odom/
python evaluate_pose.py --eval_split odom_10 --load_weights_folder ./odom_split.M/models/weights_29 --data_path kitti_odom/
```
