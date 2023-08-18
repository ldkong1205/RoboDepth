<img src="../docs/figs/logo2.png" align="right" width="34%">

# Create Corruptions

### Outline
- [Build Corruption Sets](#build-corruption-sets)
- [Corruption Taxonomy](#corruption-taxonomy)
- [Dataset Structure](#dataset-structure)
- [Simulation Algorithm]()


## Build Corruption Sets
To generate `RoboDepth` evaluation benchmarks on custom datasets, go to the `corruptions` folder:
```shell
cd corruptions/
```
Specify the corruption types to be created in an `create.sh` script, for example:
```shell
python3 corruptions/create.py \
    --image_list splits/eigen.txt \
    --H 192 \  # image height
    --W 640 \  # image width
    --save_path kitti_data/kitti_c \
    --if_zoom_blur  # create corruption type "zoom blur"
 ```

Then run `create.sh`:
```shell
sh create.sh
```

## Corruption Taxonomy

Our benchmark suite supports **18** common corruptions. We summarize the corruption types as follows:
- Weather & lighting conditions: `brightness`, `dark`, `fog`, `frost`, `snow`, and `contrast`.
- Sensor & movement: `defocus_blur`, `glass_blur`, `motion_blur`, `zoom_blur`, `elastic`, and `color_quant`.
- Data & processing: `gaussian_noise`, `impulse_noise`, `shot_noise`, `iso_noise`, `pixelate`, and `jpeg`.

<img src="../docs/figs/taxonomy.png" width="800"> |
|:-:|


## Dataset Structure
Successfully running `create.sh` will create and save images corrupted with the specified corruption types to `kitti_data/kitti_c/`. The dataset structure should end up looking like the following:
```shell
└── RoboDepth 
      │── corruptions
      │    │── create.py
      │    │── create.sh
      │    └── ...
      │── docs
      │── figs
      │── kitti_data
      │    │── 2011_09_26
      │    │── ...
      │    └── kitti_c
      │         │── brightness
      │         │── ...
      │         └── zoom_blur
      │── models
      │── zoo
      └── ...
```


## Simulation Algorithm


