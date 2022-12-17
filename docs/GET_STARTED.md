<img src="../docs/figs/logo2.png" align="right" width="34%">

# Getting Started

### Clone Repository
To begin with, clone the RoboDepth repository to the desired location:
```shell
cd models/  # desired location
```
```shell
git clone https://github.com/ldkong1205/RoboDepth.git
```

### Robustness Probing
Our benchmark currently supports the following depth estimation algorithms:

- Supervised Depth Estimation
  - [DORN](https://github.com/hufu6371/DORN)
  - [BTS](https://github.com/cleinc/bts)
  - [AdaBins](https://github.com/shariqfarooq123/AdaBins)
  - [DPT](https://github.com/isl-org/DPT)
  - [LapDepth](https://github.com/tjqansthd/LapDepth-release)
  - [SimIPU](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/simipu)
  - [NeWCRFs](https://github.com/aliyun/NeWCRFs)
  - [GLPDepth](https://github.com/vinvino02/GLPDepth)
  - [DepthFormer](https://github.com/zhyever/Monocular-Depth-Estimation-Toolbox/tree/main/configs/depthformer)

- Self-Supervised Depth Estimation
  - [MonoDepth2](https://github.com/nianticlabs/monodepth2)
  - [DepthHints](https://github.com/nianticlabs/depth-hints)
  - [FeatDepth](https://github.com/sconlyshootery/FeatDepth)
  - [PackNet-SfM](https://github.com/TRI-ML/packnet-sfm)
  - [HR-Depth](https://github.com/shawLyu/HR-Depth)
  - [ManyDepth](https://github.com/nianticlabs/manydepth)
  - [EPCDepth](https://github.com/prstrive/EPCDepth)
  - [DIFFNet](https://github.com/brandleyzhou/DIFFNet)
  - [CADepth](https://github.com/kamiLight/CADepth-master)
  - [MonoDEVSNet](https://github.com/HMRC-AEL/MonoDEVSNet)
  - [DynamicDepth](https://github.com/AutoAILab/DynamicDepth)
  - [Dyna-DM](https://github.com/kieran514/dyna-dm)

- Semi-Supervised Depth Estimation
  - [SemiDepth](https://github.com/jahaniam/semiDepth)


### MonoDepth2, ICCV'19
To evaluate the robustness of [MonoDepth2](https://github.com/nianticlabs/monodepth2) under common corruptions, run the following:
```shell
cd zoo/MonoDepth2
```
```shell
sh evaluate_kittic.sh
# --load_weights_folder: "path to the pretrained weights"
# --eval_mono or --eval_stereo: "set eval mode for mono and stereo"
```

### DIFFNet, BMVC'21
To evaluate the robustness of [DIFFNet](https://github.com/brandleyzhou/DIFFNet) under common corruptions, run the following:
```shell
cd zoo/DIFFNet
```
```shell
sh evaluate_kittic.sh
# --load_weights_folder: "path to the pretrained weights"
# --eval_mono or --eval_stereo: "set eval mode for mono and stereo"
```


