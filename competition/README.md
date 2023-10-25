<img src="../docs/figs/logo2.png" align="right" width="34%">

# RoboDepth Competition @ ICRA 2023

Welcome to the [RoboDepth Competition](https://robodepth.github.io/)! :robot:
- This is the **first** challenge on robust depth estimation under corruptions, which is associated with the 40th IEEE Conference on Robotics and Automation ([ICRA 2023](https://link.zhihu.com/?target=https%3A//www.icra2023.org/welcome)).
- In this competition, we target on probing the **Out-of-Distribution (OoD) robustness** of depth estimation models under common corruptions.
- There are **18 different corruption types** in total, ranging from three perspectives: weather and lighting conditions, sensor failure and movement, and data processing issues.
- There are **two tracks** in this competition, including self-supervised depth estimation of outdoor scenes (track 1) and fully-supervised depth estimation of indoor scenes (track 2).

<p align="center">
  <img src="../docs/figs/icra2023.png" align="center" width="100%">
</p>

This competition is sponsored by [Baidu Research](http://research.baidu.com/), USA.


## Outline
- [Useful Info](#gem-useful-info)
- [Timeline](#clock1-timeline)
- [Data Preparation](#floppy_disk-data-preparation)
- [Submission](#arrow_double_up-submission)
- [Terms & Conditions](#balance_scale-terms--conditions)
- [Organizer](#organizer)

## :gem: Useful Info
- :globe_with_meridians: - Competition page: https://robodepth.github.io.
- :wrench: - Competition toolkit: https://github.com/ldkong1205/RoboDepth.
- :oncoming_automobile: - Evaluation server (track 1): https://codalab.lisn.upsaclay.fr/competitions/9418.
- :oncoming_taxi: - Evaluation server (track 2): https://codalab.lisn.upsaclay.fr/competitions/9821.
- :octocat: - Official GitHub account: https://github.com/RoboDepth.
- :mailbox: - Contact: robodepth@outlook.com.


## :clock1: Timeline
- \[2023.01.01\] - Competition launches
- \[2023.01.02\] - Track 1 (self-supervised depth estimation) starts
- \[2023.01.15\] - Track 2 (fully-supervised depth estimation) starts
- \[2023.05.25\] - Competition ends
- \[2023.05.29\] - Workshop & discussion
- \[2023.06.02\] - Release of results @ ICRA 2023


## :floppy_disk: Data Preparation

### \[Track 1\]: Self-Supervised Depth Estimation

### :hamster: Training Set
> In this track, the participants are expected to adopt the data from the [raw KITTI dataset](http://www.cvlibs.net/datasets/kitti/raw_data.php) for model training. You can download this dataset by running:
> ```shell
> wget -i splits/kitti_archives_to_download.txt -P kitti_data/
> ```
> Then unzip with:
> ```shell
> cd kitti_data/
> unzip "*.zip"
> cd ..
> ```
> Please note that this dataset weighs about `175GB`, so make sure you have enough space to `unzip` too!

> The **training split** of this dataset is defined in the `splits/` folder of this codebase. By default, we **require** all participants to train their depth estimation models using Zhou's subset of the standard **Eigen split** of KITTI, which is designed for self-supervised monocular training.

> :warning: Regarding the **data augmentation** to be adopted during the training phase, please refer to the [Terms & Conditions](#balance_scale-terms--conditions) section.

### :robot: Evaluation Set
> In this track, the participants are expected to adopt our generated data for model evaluation. There are multiple ways of accessing this evaluation set. In particular, you can download the data from Google Drive via the following link:<br>
> :link: https://drive.google.com/file/d/14Z0k2lhpk0D0pkyzIcHyk4Ce0wS3IcfF/view?usp=sharing.

> Alternatively, you can download the data from [this](https://codalab.lisn.upsaclay.fr/competitions/9418#participate-get_starting_kit) CodaLab page. Please note that you need to **register** for this track first before entering the downloading page.

> This evaluation set weighs about `100MB`. It includes **500 corrupted images**, generated under the mentioned 18 corruption types. In this competition, we will evaluate the model performance using the ground-truth depth of these images. The participants are required to submit the prediction file to [this](https://codalab.lisn.upsaclay.fr/competitions/9418) evaluation server. For more details on the submission, please refer to the [Submission](#arrow_double_up-submission) section.

<hr>

### \[Track 2\]: Fully-Supervised Depth Estimation

### :hamster: Training Set
> In this track, the participants are expected to adopt the data from the [NYU Depth Dataset V2](https://cs.nyu.edu/~silberman/datasets/nyu_depth_v2.html) for model training. You can download this dataset from Google Drive with the following link:<br>
> :link: https://drive.google.com/file/d/1wC-io-14RCIL4XTUrQLk6lBqU2AexLVp/view?usp=sharing.

> Alternatively, you can download the data to the server by running:
> ```shell
> wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1wC-io-14RCIL4XTUrQLk6lBqU2AexLVp' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1wC-io-14RCIL4XTUrQLk6lBqU2AexLVp" -O nyu.zip && rm -rf /tmp/cookies.txt
> ```
> Then unzip with:
> ```shell
> unzip nyu.zip
> ```

> :warning: Regarding the **data augmentation** to be adopted during the training phase, please refer to the [Terms & Conditions](#balance_scale-terms--conditions) section.


### :robot: Evaluation Set
> In this track, the participants are expected to adopt our generated data for model evaluation. There are multiple ways of accessing this evaluation set. In particular, you can download the data from Google Drive via the following link:<br>
> :link: https://drive.google.com/file/d/1HIJxmNBFaHwSUABnkEgnFm9EdBBgvozP/view?usp=sharing.

> Alternatively, you can download the data from [this](https://codalab.lisn.upsaclay.fr/competitions/9821#participate-get_starting_kit) CodaLab page. Please note that you need to **register** for this track first before entering the downloading page.

> This evaluation set weighs about `12MB`. It includes **200 corrupted images**, generated under the mentioned 15 corruption types (excluded: `fog`, `frost`, and `snow`). In this competition, we will evaluate the model performance using the ground-truth depth of these images. The participants are required to submit the prediction file to [this](https://codalab.lisn.upsaclay.fr/competitions/9821) evaluation server. For more details on the submission, please refer to the [Submission](#arrow_double_up-submission) section.


## :arrow_double_up: Submission

### \[Track 1\]: Self-Supervised Depth Estimation
> In this track, the participants are expected to submit their predictions to the CodaLab server for model evaluation. Specifically, you can access the server of this track via the following link:<br>
> :link: https://codalab.lisn.upsaclay.fr/competitions/9418.

> In order to make a successful submission and evaluation, you need to follow these instructions:

> **\[Registration\]**<br> You will need to **register for this track** on CodaLab before you can make a submission. To achieve this, apply for a CodaLab account if you do not have one, with your email. Then, go to the server page of this track and press `Participate`; you will see a `Sign In` button. Click it for registration.

> **\[File Preparation\]**<br> You will need to prepare the **model prediction file** for submission. Specifically, the evaluation server of this track accepts the `.zip` file of your model predictions in `numpy array` format. You can follow the example below, which is modified based on the evaluation code from [MonoDepth2](https://arxiv.org/abs/1806.01260):<br>
> - Step 1: Generate your model predictions with: 
>   ```shell
>   pred_disps = []
>   
>   with torch.no_grad():
>       for data in dataloader:
>           input_color = data[("color", 0, -1)].to(device)
>           output = depth_decoder(encoder(input_color))
>           pred_disp, _ = disp_to_depth(output[("disp", 0)], opt.min_depth, opt.max_depth)
>           pred_disp = pred_disp.cpu()[:, 0].numpy()
>           pred_disps.append(pred_disp)
>   ```   
> - Step 2: After evaluating every sample in the evaluation set, save the prediction file with:
>   ```shell
>   output_path = os.path.join(opt.save_pred_path, "disp.npy")
>   np.save(output_path, pred_disps)
>   ```
> - Step 3: Compress the saved `.npy` file with:
>   ```shell
>   zip disp.zip disp.npy
>   ```
> - Step 4: Download `disp.zip` from your computing machine.

> **\[Submission & Evaluation\]**<br> You will need to submit your `disp.zip` file manually to the evaluation server. To achieve this, go to the server page of this track and press `Participate`; you will see a `Submit / View Results` button. Click it for submission. You are encouraged to fill in the submission info with your *team name*, *method name*, and *method description*. Then, click the `Submit` button and select your `disp.zip` file. After successfully uploading the file, the server will automatically evaluate the performance of your submission and put the results on the leaderboard.<br>
> :warning: Do not close the page when you are uploading the prediction file.

> **\[View Result\]**<br> You can view your scores by pressing the `Results` button. Following the same configuration with [MonoDepth2](https://arxiv.org/abs/1806.01260), we evaluate the model performance with 7 metrics: `abs_rel`, `sq_rel`, `rmse`, `rmse_log`, `a1`, `a2`, and `a3`.

<hr>

### \[Track 2\]: Fully-Supervised Depth Estimation
> In this track, the participants are expected to submit their predictions to the CodaLab server for model evaluation. Specifically, you can access the server of this track via the following link:<br>
> :link: https://codalab.lisn.upsaclay.fr/competitions/9821.

> In order to make a successful submission and evaluation, you need to follow these instructions:

> **\[Registration\]**<br> You will need to **register for this track** on CodaLab before you can make a submission. To achieve this, apply for a CodaLab account if you do not have one, with your email. Then, go to the server page of this track and press `Participate`; you will see a `Sign In` button. Click it for registration.

> **\[File Preparation\]**<br> You will need to prepare the **model prediction file** for submission. Specifically, the evaluation server of this track accepts the `.zip` file of your model predictions in `numpy array` format. You can follow the example below, which is modified based on the evaluation code from the [Monocular-Depth-Estimation-Toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox):<br>
> - Step 1: Generate your model predictions with: 
>   ```shell
>   pred_disps = []
>
>   for batch_indices, data in zip(loader_indices, data_loader):
>       with torch.no_grad():
>           result = model(return_loss=False, rescale=True, **data)
>           pred_disps.append(result)
>   ```
>   Please notice that you will need to sort the file paths before inference. As been pointed out by @Zhyever in [this](https://github.com/ldkong1205/RoboDepth/issues/10) issue, you can achieve this via the following line of code:
>   ```
>   img_infos = sorted(img_infos, key=lambda x: x['filename'])
>   ```
> - Step 2: After evaluating every sample in the evaluation set, save the prediction file with:
>   ```shell
>   output_path = os.path.join(opt.save_pred_path, "disp.npz")
>   np.savez_compressed(output_path, data=pred_disps)
>   ```
> - Step 3: Compress the saved `.npz` file with:
>   ```shell
>   zip disp.zip disp.npz
>   ```
> - Step 4: Download `disp.zip` from your computing machine.

> **\[Submission & Evaluation\]**<br> You will need to submit your `disp.zip` file manually to the evaluation server. To achieve this, go to the server page of this track and press `Participate`; you will see a `Submit / View Results` button. Click it for submission. You are encouraged to fill in the submission info with your *team name*, *method name*, and *method description*. Then, click the `Submit` button and select your `disp.zip` file. After successfully uploading the file, the server will automatically evaluate the performance of your submission and put the results on the leaderboard.<br>
> :warning: Do not close the page when you are uploading the prediction file.

> **\[View Result\]**<br> You can view your scores by pressing the `Results` button. Following the same configuration in the [Monocular-Depth-Estimation-Toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox), we evaluate the model performance with 9 metrics: `a1`, `a2`, `a3`, `abs_rel`, `sq_rel`, `rmse`, `rmse_log`, `log10`, and `silog`.



## :balance_scale: Terms & Conditions
This competition is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:
1. That the data in this competition comes “AS IS”, without express or implied warranty. Although every effort has been made to ensure accuracy, we do not accept any responsibility for errors or omissions.
2. That you may not use the data in this competition or any derivative work for commercial purposes such as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
3. That you include a reference to RoboDepth (including the benchmark data and the specially generated data for academic challenges) in any work that makes use of the benchmark. For research papers, please cite our preferred publications as listed on our webpage.

To ensure a **fair comparison** among all participants, we require:
1. All participants must follow the **exact same data configuration** when training and evaluating their algorithms. Please do not use any public or private datasets other than those specified for model training.
2. The theme of this competition is to probe the out-of-distribution robustness of depth estimation models. Therefore, any use of the 18 corruption types designed in this benchmark is **strictly prohibited**, including any atomic operation that comprises any one of the mentioned corruptions.
    - For Track 1: Please stick with the default data augmentations used in the [MonoDepth2](https://github.com/nianticlabs/monodepth2) codebase.
    - For Track 2: Please stick with the default data augmentations used in the [Monocular-Depth-Estimation-Toolbox](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox) codebase.
4. To ensure the above two rules are followed, each participant is requested to **submit the code** with reproducible results before the final result is announced; the code is for examination purposes only and we will manually verify the training and evaluation of each participant's model.

:blush: If you have any questions or concerns, please get in touch with us at robodepth@outlook.com.


## Organizer
<p align="center">
  <img src="../docs/figs/icra2023_organizer.png" align="center" width="100%">
</p>
