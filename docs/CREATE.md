<img src="../docs/figs/logo2.png" align="right" width="34%">

# Create Corruptions

### Outline
- [Build Corruption Sets](#build-corruption-sets)
- [Corruption Taxonomy](#corruption-taxonomy)
- [Dataset Structure](#dataset-structure)
- [Simulation Algorithm](#simulation-algorithm)
  - [Configuration Details](#configuration-details)
  - [Brightness](#brightness)
- [Acknowledgement](#acknowledgement)


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

### Configuration Details

| Corruption | Parameter | Level 1 | Level 2 | Level 3 | Level 4 | Level 5 | 
|:-|:-:|:-:|:-:|:-:|:-:|:-:|
| `brightness` | adjustment in HSV space | 0.1 | 0.2 | 0.3 | 0.4 | 0.5 |
| `dark` | 
| `fog` | (thickness, smoothness) | (1.5, 2.0) | (2.0, 2.0) | (2.5, 1.7) | (2.5, 1.5) | (3.0, 1.4) |
| `frost` |
| `snow` |
| `contrast` |



### Brightness
The `brightness` function alters the HSV (Hue, Saturation, Value) color space of an image, adjusting the brightness component. Specifically, it simulates brightness changes by adding or subtracting an adjustment coefficient, resulting in an overall brightening or darkening effect on the image.

```python
def brightness(x, severity=1):
    c = [.1, .2, .3, .4, .5][severity - 1]

    x = np.array(x) / 255.

    if len(x.shape) < 3 or x.shape[2] < 3:
        x = np.clip(x + c, 0, 1)
    else:
        x = sk.color.rgb2hsv(x)
        x[:, :, 2] = np.clip(x[:, :, 2] + c, 0, 1)
        x = sk.color.hsv2rgb(x)

    return np.clip(x, 0, 1) * 255
```


### Fog
The `fog` function applies a simulated fog effect to an image by adding a foggy texture generated through plasma fractals. The fog effect is controlled by parameters that determine the thickness and smoothness of the fog, creating a visual distortion resembling the appearance of fog.

```python
def fog(x, severity=1):
    c = [(1.5, 2), (2., 2), (2.5, 1.7), (2.5, 1.5), (3., 1.4)][severity - 1]

    shape = np.array(x).shape
    max_side = np.max(shape)
    map_size = next_power_of_2(int(max_side))

    x = np.array(x) / 255.
    max_val = x.max()

    x_shape = np.array(x).shape
    if len(x_shape) < 3 or x_shape[2] < 3:
        x += c[0] * plasma_fractal(mapsize=map_size, wibbledecay=c[1])[
                    :shape[0], :shape[1]]
    else:
        x += c[0] * \
             plasma_fractal(mapsize=map_size, wibbledecay=c[1])[:shape[0],
             :shape[1]][..., np.newaxis]

    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
```


## Acknowledgement


