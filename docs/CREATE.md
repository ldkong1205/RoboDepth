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
| `frost` | (frost intensity, texture influence) | (1.00, 0.40) | (0.80, 0.60) | (0.70, 0.70) | (0.65, 0.70) | (0.60, 0.75) |
| `snow` | (mean, std, scale, threshold, blur radius, blur std, blending ratio) | (0.1, 0.3, 3.0, 0.5, 10.0, 4.0, 0.8) | (0.2, 0.3, 2, 0.5, 12, 4, 0.7) | (0.55, 0.3, 4, 0.9, 12, 8, 0.7) | (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65) | (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55) |
| `contrast` | adjustment of pixel mean | 0.40 | 0.30 | 0.20 | 0.10 | 0.05 |



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


### Dark



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
        x += c[0] * plasma_fractal(mapsize=map_size, wibbledecay=c[1])[:shape[0], :shape[1]]
    else:
        x += c[0] * plasma_fractal(mapsize=map_size, wibbledecay=c[1])[:shape[0], :shape[1]][..., np.newaxis]

    return np.clip(x * max_val / (max_val + c[0]), 0, 1) * 255
```


### Frost
The `frost` function applies a simulated frost effect to an image by overlaying frost textures obtained from pre-defined frost images. The frost effect introduces icy patterns to the image, creating the appearance of frost accumulation.

```python
def frost(x, severity=1):
    c = [(1, 0.4), (0.8, 0.6), (0.7, 0.7), (0.65, 0.7), (0.6, 0.75)][severity - 1]

    idx = np.random.randint(5)
    filename = [resource_filename(__name__, './frost/frost1.png'), resource_filename(__name__, './frost/frost2.png'), resource_filename(__name__, './frost/frost3.png'), resource_filename(__name__, './frost/frost4.jpg'), resource_filename(__name__, './frost/frost5.jpg'), resource_filename(__name__, './frost/frost6.jpg')][idx]
    frost = cv2.imread(filename)
    frost_shape = frost.shape
    x_shape = np.array(x).shape

    scaling_factor = 1
    if frost_shape[0] >= x_shape[0] and frost_shape[1] >= x_shape[1]:
        scaling_factor = 1
    elif frost_shape[0] < x_shape[0] and frost_shape[1] >= x_shape[1]:
        scaling_factor = x_shape[0] / frost_shape[0]
    elif frost_shape[0] >= x_shape[0] and frost_shape[1] < x_shape[1]:
        scaling_factor = x_shape[1] / frost_shape[1]
    elif frost_shape[0] < x_shape[0] and frost_shape[1] < x_shape[1]:
        scaling_factor_0 = x_shape[0] / frost_shape[0]
        scaling_factor_1 = x_shape[1] / frost_shape[1]
        scaling_factor = np.maximum(scaling_factor_0, scaling_factor_1)

    scaling_factor *= 1.1
    new_shape = (int(np.ceil(frost_shape[1] * scaling_factor)), int(np.ceil(frost_shape[0] * scaling_factor)))
    frost_rescaled = cv2.resize(frost, dsize=new_shape, interpolation=cv2.INTER_CUBIC)
    x_start, y_start = np.random.randint(0, frost_rescaled.shape[0] - x_shape[0]), np.random.randint(0, frost_rescaled.shape[1] - x_shape[1])

    if len(x_shape) < 3 or x_shape[2] < 3:
        frost_rescaled = frost_rescaled[x_start:x_start + x_shape[0], y_start:y_start + x_shape[1]]
        frost_rescaled = rgb2gray(frost_rescaled)
    else:
        frost_rescaled = frost_rescaled[x_start:x_start + x_shape[0], y_start:y_start + x_shape[1]][..., [2, 1, 0]]

    return np.clip(c[0] * np.array(x) + c[1] * frost_rescaled, 0, 255)
```

### Snow
The `snow` function simulates a snow effect on an image by generating a layer of snow-like particles and adding motion blur to create the appearance of falling snowflakes. The snow particles are combined with the input image, introducing a snowy ambiance.

```python
def snow(x, severity=1):
    c = [(0.1, 0.3, 3, 0.5, 10, 4, 0.8), (0.2, 0.3, 2, 0.5, 12, 4, 0.7), (0.55, 0.3, 4, 0.9, 12, 8, 0.7), (0.55, 0.3, 4.5, 0.85, 12, 8, 0.65), (0.55, 0.3, 2.5, 0.85, 12, 12, 0.55)][severity - 1]

    x = np.array(x, dtype=np.float32) / 255.
    snow_layer = np.random.normal(size=x.shape[:2], loc=c[0], scale=c[1])

    snow_layer = clipped_zoom(snow_layer[..., np.newaxis], c[2])
    snow_layer[snow_layer < c[3]] = 0

    snow_layer = np.clip(snow_layer.squeeze(), 0, 1)
    snow_layer = _motion_blur(snow_layer, radius=c[4], sigma=c[5], angle=np.random.uniform(-135, -45))
    snow_layer = np.round(snow_layer * 255).astype(np.uint8) / 255.
    snow_layer = snow_layer[..., np.newaxis]
    snow_layer = snow_layer[:x.shape[0], :x.shape[1], :]

    if len(x.shape) < 3 or x.shape[2] < 3:
        x = c[6] * x + (1 - c[6]) * np.maximum(x, x.reshape(x.shape[0], x.shape[1]) * 1.5 + 0.5)
        snow_layer = snow_layer.squeeze(-1)
    else:
        x = c[6] * x + (1 - c[6]) * np.maximum(x, cv2.cvtColor(x, cv2.COLOR_RGB2GRAY).reshape(x.shape[0], x.shape[1], 1) * 1.5 + 0.5)
    try:
        return np.clip(x + snow_layer + np.rot90(snow_layer, k=2), 0, 1) * 255
    except ValueError:
        print('ValueError for Snow, Exception handling')
        x[:snow_layer.shape[0], :snow_layer.shape[1]] += snow_layer + np.rot90(snow_layer, k=2)
            
        return np.clip(x, 0, 1) * 255
```

### Contrast
The `contrast` function modifies the contrast of an image by adjusting pixel values around their mean. The degree of adjustment is controlled by a parameter, resulting in enhanced or reduced contrast.

```python
def contrast(x, severity=1):
    c = [0.4, .3, .2, .1, .05][severity - 1]

    x = np.array(x) / 255.
    means = np.mean(x, axis=(0, 1), keepdims=True)
    return np.clip((x - means) * c + means, 0, 1) * 255
```




## Acknowledgement


