# MonoViT

This is the reference PyTorch implementation for training and testing depth estimation models using the method described in

> **MonoViT: Self-Supervised Monocular Depth Estimation with a Vision Transformer** [arxiv](https://arxiv.org/abs/2208.03543)
> 
>Chaoqiang Zhao*, Youmin Zhang*, Matteo Poggi, Fabio Tosi, Xianda Guo,Zheng Zhu, Guan Huang, Yang Tang, Stefano Mattoccia

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/monovit-self-supervised-monocular-depth/monocular-depth-estimation-on-kitti-eigen-1)](https://paperswithcode.com/sota/monocular-depth-estimation-on-kitti-eigen-1?p=monovit-self-supervised-monocular-depth)

<div class='paper-box'><div class='paper-box-image'><img src='fig/kittiandds.png' alt="sym" width="90%"></div>
<div class='paper-box-text' markdown="1">


If you find our work useful in your research please consider citing our paper:

```
@inproceedings{monovit,
  title={MonoViT: Self-Supervised Monocular Depth Estimation with a Vision Transformer},
  author={Zhao, Chaoqiang and Zhang, Youmin and Poggi, Matteo and Tosi, Fabio and Guo, Xianda and Zhu, Zheng and Huang, Guan and Tang, Yang and Mattoccia, Stefano},
  booktitle={International Conference on 3D Vision},
  year={2022}
}
```



## ⚙️ Setup

Assuming a fresh [Anaconda](https://www.anaconda.com/download/) distribution, you can install the dependencies with:
```shell
pip3 install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0
pip install dominate==2.4.0 Pillow==6.1.0 visdom==0.1.8
pip install tensorboardX==1.4 opencv-python  matplotlib scikit-image
pip3 install mmcv-full==1.3.0 mmsegmentation==0.11.0  
pip install timm einops IPython
```
We ran our experiments with PyTorch 1.9.0, CUDA 11.1, Python 3.7 and Ubuntu 18.04.

Note that our code is built based on [Monodepth2](https://github.com/nianticlabs/monodepth2)

## Results on KITTI

We provide the following  options for `--model_name`:

| `--model_name`          | Training modality | Pretrained? | Model resolution  |Abs Rel| Sq Rel| RMSE| RMSE log|  delta < 1.25  | delta < 1.25^2  | delta < 1.25^3  |
|-----------------------|-------------|------|-----------------|----|----|----|------|--------|--------|--------|
| [`mono_640x192`](https://drive.google.com/drive/folders/1VWDPuqiMPDD2P--Oka-yJgh8z7ouCX4D?usp=sharing)          | Mono              | Yes | 640 x 192                | 0.099 |0.708 |4.372| 0.175 |0.900 |0.967| 0.984|
| [`mono+stereo_640x192`](https://drive.google.com/drive/folders/1_HPsL1Vg3s0LdOykfTT0aMlE6-u3IxQn?usp=sharing)   | Mono + Stereo     | Yes | 640 x 192                | 0.098| 0.683| 4.333| 0.174| 0.904| 0.967| 0.984|
| [`mono_1024x320`](https://drive.google.com/drive/folders/1EDTSZ59CGW9rUoDL3EwEKn3PpZpUUGsS?usp=sharing)         | Mono              | Yes | 1024 x 320               | 0.096|  0.714|  4.292|  0.172|  0.908|  0.968|  0.984|
| [`mono+stereo_1024x320`](https://drive.google.com/drive/folders/1tez1RQFO33MMyVAq_gkOVHoL2TO98-TH?usp=sharing)  | Mono + Stereo     | Yes | 1024 x 320               | 0.093 |0.671 |4.202 |0.169 |0.912 |0.969 |0.985|
| [`mono_1280x384`](https://drive.google.com/drive/folders/1l3egRvLaoBqgYrgfktgpJt613QwZ4twT?usp=sharing)         | Mono              | Yes | 1280 x 384               | 0.094 |0.682| 4.200| 0.170| 0.912| 0.969| 0.984|



## 💾 KITTI training data

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

The above conversion command creates images which match our experiments, where KITTI `.png` images were converted to `.jpg` on Ubuntu 16.04 with default chroma subsampling `2x2,1x1,1x1`.
We found that Ubuntu 18.04 defaults to `2x2,2x2,2x2`, which gives different results, hence the explicit parameter in the conversion command.

You can also place the KITTI dataset wherever you like and point towards it with the `--data_path` flag during training and evaluation.

**Splits**

The train/test/validation splits are defined in the `splits/` folder.
By default, the code will train a depth model using [Zhou's subset](https://github.com/tinghuiz/SfMLearner) of the standard Eigen split of KITTI, which is designed for monocular training.
You can also train a model using the new [benchmark split](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) or the [odometry split](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) by setting the `--split` flag.


**Custom dataset**

You can train on a custom monocular or stereo dataset by writing a new dataloader class which inherits from `MonoDataset` – see the `KITTIDataset` class in `datasets/kitti_dataset.py` for an example.


## ⏳ Training

PLease download the ImageNet-1K pretrained MPViT [model](https://dl.dropbox.com/s/y3dnmmy8h4npz7a/mpvit_small.pth) to `./ckpt/`.

For training, please download monodepth2, replace the depth network, and revise the setting of the depth network, the optimizer and learning rate according to `trainer.py`. 

Because of the different torch version between MonoViT and Monodepth2, the func `transforms.ColorJitter.get_params` in dataloader should also be revised to `transforms.ColorJitter`.

By default models and tensorboard event files are saved to `./tmp/<model_name>`.
This can be changed with the `--log_dir` flag.


**Monocular training:**
```shell
python train.py --model_name mono_model --learning_rate 5e-5
```

**Monocular + stereo training:**
```shell
python train.py --model_name mono+stereo_model --use_stereo --learning_rate 5e-5
```


### GPUs

The code of the Single GPU version can only be run on a single GPU.
You can specify which GPU to use with the `CUDA_VISIBLE_DEVICES` environment variable:
```shell
CUDA_VISIBLE_DEVICES=1 python train.py --model_name mono_model
```

## 📊 KITTI evaluation

To prepare the ground truth depth maps, please follow the monodepth2.

...assuming that you have placed the KITTI dataset in the default location of `./kitti_data/`.

The following example command evaluates the epoch 19 weights of a model named `mono_model` (Note that please use `evaluate_depth.py` for 640x192 models and `evaluate_hr_depth.py --height 320/384 --width 1024/1280` for the others):
```shell
python evaluate_depth.py --load_weights_folder ./tmp/mono_model/models/weights_19/ --eval_mono
```


An additional parameter `--eval_split` can be set.
The three different values possible for `eval_split` are explained here:

| `--eval_split`        | Test set size | For models trained with... | Description  |
|-----------------------|---------------|----------------------------|--------------|
| **`eigen`**           | 697           | `--split eigen_zhou` (default) or `--split eigen_full` | The standard Eigen test files |
| **`eigen_benchmark`** | 652           | `--split eigen_zhou` (default) or `--split eigen_full`  | Evaluate with the improved ground truth from the [new KITTI depth benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) |
| **`benchmark`**       | 500           | `--split benchmark`        | The [new KITTI depth benchmark](http://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction) test files. |

## Contact us

Contact us: zhaocqilc@gmail.com

## Acknowledgement
Thanks the authors for their works:

[Monodepth2](https://github.com/nianticlabs/monodepth2)

[MPVIT](https://github.com/youngwanLEE/MPViT)

[HR-Depth](https://github.com/shawLyu/HR-Depth)

[DIFFNet](https://github.com/brandleyzhou/DIFFNet)
