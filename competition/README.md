<img src="../docs/figs/logo2.png" align="right" width="34%">

# RoboDepth Competition @ ICRA 2023

Welcome to the [RoboDepth Competition](https://robodepth.github.io/)! :robot:
- This is the **first** challenge on robust depth estimation under corruptions, which is associated with the 40th IEEE Conference on Robotics and Automation ([ICRA 2023](https://link.zhihu.com/?target=https%3A//www.icra2023.org/welcome)).
- In this competition, we target on probing the **Out-of-Distribution (OoD) robustness** of depth estimation models under common corruptions.
- There are **18 different corruption types** in total, ranging from three perspectives: weather and lighting conditions, sensor failure and movement, and data processing issues.
- There are **two tracks** in this competition, including self-supervised depth estimation of outdoor scenes (track 1) and fully-supervised depth estimation of indoor scenes (track 2).

<p align="center">
  <img src="../docs/figs/icra2023.jpeg" align="center" width="100%">
</p>


## Outline
- [Useful Info](#useful-info)
- [Timeline](#timeline)
- [Data Preparation](#data-preparation)
- [Submission](#submission)
- [Terms & Conditions](#terms--conditions)


## Useful Info
- :globe_with_meridians: Competition page: https://robodepth.github.io.
- :wrench: Competition toolkit: https://github.com/ldkong1205/RoboDepth.
- :oncoming_automobile: Evaluation server (track 1): https://codalab.lisn.upsaclay.fr/competitions/9418.
- :oncoming_taxi: Evaluation server (track 1): Coming soon.
- :octocat: Official GitHub account: https://github.com/RoboDepth.
- :mailbox: Contact: robodepth@outlook.com.


## Timeline
- \[2023.01.01\] - Competition launches
- \[2023.01.02\] - Track 1 (self-supervised depth estimation) starts
- \[2023.01.15\] - Track 2 (fully-supervised depth estimation) starts
- \[2023.05.25\] - Competition ends
- \[2023.05.29\] - Workshop & discussion
- \[2023.06.02\] - Release of results @ ICRA 2023


## Data Preparation

### Track 1: Self-Supervised Depth Estimation

### :hamster: Training Set
In this track, the participants are expected to adopt the data from the [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) for model training. You can download this dataset by running:
```shell
wget -i splits/kitti_archives_to_download.txt -P kitti_data/
```

Then unzip with:
```shell
cd kitti_data/
unzip "*.zip"
cd ..
```

Please note that this dataset weighs about `175GB`, so make sure you have enough space to `unzip` too!

The **training split** of this dataset is defined in the `splits/` folder of this codebase. By default, we **require** all participants to train their depth estimation models using Zhou's subset of the standard Eigen split of KITTI, which is designed for self-supervised monocular training.

:warning: Regarding the **data augmentation** to be adopted during the training phase, please refer to the rules in the [Terms & Conditions](#terms--conditions) section.

### :robot: Evaluation Set
In this track, the participants are expected to adopt our generated data model evaluation. There are multiple ways of accessing the evaluation set. In particular, you can download the data from Google Drive via the following link:

:link: https://drive.google.com/file/d/14Z0k2lhpk0D0pkyzIcHyk4Ce0wS3IcfF/view?usp=sharing

Alternatively, you can download the data from [this](https://codalab.lisn.upsaclay.fr/competitions/9418#participate-get_starting_kit) CodaLab page. Please note that you need to register for this track first before entering the downloading page.

This evaluation set weighs about `100MB`. It includes **500 corrupted images**, generated under the mentioned 18 corruption types. In this competition, we will evaluate the model performance using the ground-truth depth of these images. The participants are required to submit the prediction file to this [evaluation server](https://codalab.lisn.upsaclay.fr/competitions/9418). For more details on the submission, please refer to the [Submission](#submission) section.

<hr>

### Track 2: Fully-Supervised Depth Estimation

### :hamster: Training Set
Coming soon!

### :robot: Evaluation Set
Coming soon!


## Submission

## Terms & Conditions


