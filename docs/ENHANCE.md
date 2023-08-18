<img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/logo2.png" align="right" width="34%">

# Robustness Enhancement

### Outline
- [Train & Test on Clean Data](#train--test-on-clean-data)
- [Train & Test on Corrupted Data](#train--test-on-corrupted-data)


## Train & Test on Clean Data

### Train on Clean, Test on Clean
> **Model:** MonoDepth2, **Backbone:** ResNet-18

| Train | Abs Rel | Sq Rel | RMSE | RMSE log | a1 | a2 | a3 | DEE | mCE (%) | mRR (%) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Clean | 0.115	| 0.905	| 4.864 |	0.193	| 0.877 |	0.959	| 0.981 | 0.119 | 100.00 | 84.46 |


### Train on Corruptions, Test on Clean
> **Model:** MonoDepth2, **Backbone:** ResNet-18, **Corrupt Probability:** 1.0

| Train | Abs Rel | Sq Rel | RMSE | RMSE log | a1 | a2 | a3 | DEE | mCE (%) | mRR (%) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Bright   | 0.120 | 0.951 | 4.953 | 0.197 | 0.866 | 0.956 | 0.980 | 0.127 | 101.98 | 85.06 |
| Dark     | 0.130 | 1.082 | 5.251 | 0.206 | 0.848 | 0.952 | 0.979 | 0.141 | 111.11 | 84.90 |
| Fog      | 0.119 | 0.883 | 4.972 | 0.197 | 0.865 | 0.957 | 0.981 | 0.127 | 106.54 | 82.91 |
| Snow     | 0.123 | 0.920 | 4.987 | 0.200 | 0.859 | 0.955 | 0.980 | 0.132 | 102.07 | 86.35 |
| Contrast | 0.119 | 0.899 | 4.994 | 0.199 | 0.865 | 0.955 | 0.980 | 0.127 | 111.38 | 81.49 |
| Defocus  | 0.148 | 1.138 | 6.238 | 0.245 | 0.796 | 0.925 | 0.966 | 0.176 | 114.57 | 85.80 |
| Motion   | 0.124 | 0.909 | 5.067 | 0.202 | 0.855 | 0.952 | 0.979 | 0.135 | 100.78 | 85.87 |
| Zoom     | 0.160 | 1.191 | 6.023 | 0.244 | 0.768 | 0.923 | 0.969 | 0.196 | 132.44 | 82.84 |
| Elastic  | 0.122 | 0.950 | 5.015 | 0.200 | 0.865 | 0.955 | 0.979 | 0.129 | 110.40 | 81.94 |
| Gaussian | 0.126 | 1.023 | 5.091 | 0.203 | 0.856 | 0.953 | 0.979 | 0.135 | 94.26  | 90.05 |
| Impulse  | 0.126 | 1.025 | 5.118 | 0.203 | 0.857 | 0.953 | 0.979 | 0.135 | 94.16  | 90.06 |
| Shot     | 0.124 | 0.988 | 5.028 | 0.200 | 0.860 | 0.955 | 0.980 | 0.132 | 90.12  | 90.64 |
| ISO      | 0.127 | 0.984 | 5.055 | 0.203 | 0.853 | 0.953 | 0.980 | 0.137 | 102.64 | 87.93 |
| Pixelate | 0.122 | 0.938 | 4.978 | 0.199 | 0.862 | 0.955 | 0.980 | 0.130 | 106.78 | 83.35 |
| JPEG     | 0.125 | 0.973 | 5.034 | 0.202 | 0.859 | 0.954 | 0.979 | 0.133 | 106.33 | 85.06 |
||
| Combo (W&L) | 0.120 | 0.924 | 4.914 | 0.196 | 0.866 | 0.957 | 0.980 | 0.127 | 83.75 | 91.93 |
| Combo (S&M) | 0.128 | 1.009 | 5.132 | 0.205 | 0.850 | 0.952 | 0.979 | 0.139 | 93.94 | 89.15 |
| Combo (D&P) | 0.124 | 0.979 | 5.064 | 0.201 | 0.858 | 0.954 | 0.980 | 0.133 | 90.44 | 91.02 |
||
| Combo (All) | 0.124 | 0.963 | 5.040 | 0.199 | 0.860 | 0.955 | 0.980 | 0.132 | 71.25 | 96.75 |

> **Model:** MonoDepth2, **Backbone:** ResNet-18, **Corrupt Probability:** 0.5

| Train | Abs Rel | Sq Rel | RMSE | RMSE log | a1 | a2 | a3 | DEE | mCE (%) | mRR (%) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Bright   |
| Dark     |
| Fog      |
| Snow     |
| Contrast |
| Defocus  |
| Motion   |
| Zoom     |
| Elastic  |
| Gaussian |
| Impulse  |
| Shot     |
| ISO      |
| Pixelate |
| JPEG     |
||
| Combo (W&L) | 0.120 | 0.945 | 4.925 | 0.195 | 0.868 | 0.957 | 0.980 |
| Combo (S&M) | 0.122 | 0.961 | 5.026 | 0.199 | 0.864 | 0.956 | 0.980 |
| Combo (D&P) | 0.120 | 0.912 | 4.951 | 0.198 | 0.865 | 0.956 | 0.980 |
||
| Combo (All) | 0.118 | 0.912 | 4.935 | 0.195 | 0.867 | 0.957 | 0.981 |

> **Model:** MonoDepth2, **Backbone:** ResNet-18, **Corrupt Probability:** 0.25

| Train | Abs Rel | Sq Rel | RMSE | RMSE log | a1 | a2 | a3 | DEE | mCE (%) | mRR (%) |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Bright   | 
| Dark     |
| Fog      |
| Snow     |
| Contrast |
| Defocus  |
| Motion   |
| Zoom     |
| Elastic  |
| Gaussian |
| Impulse  |
| Shot     |
| ISO      |
| Pixelate |
| JPEG     |
||
| Combo (W&L) | 0.118 | 0.925 | 4.882 | 0.194 | 0.870 | 0.958 | 0.981 |
| Combo (S&M) | 0.120 | 0.936 | 4.925 | 0.196 | 0.866 | 0.956 | 0.980 |
| Combo (D&P) | 0.118 | 0.904 | 4.920 | 0.197 | 0.868 | 0.956 | 0.980 |
| Combo (All) | 0.117 | 0.907 | 4.861 | 0.195 | 0.872 | 0.958 | 0.980 |


## Train & Test on Corrupted Data

### Train on Clean, Test on Corruptions
> **Model:** MonoDepth2, **Backbone:** ResNet-18, **Metric:** Abs Rel

| Train | Bright | Dark | Fog | Frost | Snow | Contrast | Defocus | Glass | Motion | Zoom | Elastic| Quant| Gaussian | Impulse | Shot | ISO | Pixelate | JPEG |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Clean |

### Train on Corruptions, Test on Corruptions
> **Model:** MonoDepth2, **Backbone:** ResNet-18, **Corrupt Probability:** 1.0, **Metric:** Abs Rel

| Train | Bright | Dark | Fog | Frost | Snow | Contrast | Defocus | Glass | Motion | Zoom | Elastic| Quant| Gaussian | Impulse | Shot | ISO | Pixelate | JPEG |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Bright   |.118|.223|.139|.218|.377|.164|.222|.218|.190|.172|.127|.167|.269|.285|.261|.271|.136|.160
| Dark     |
| Fog      |
| Snow     |
| Contrast |
| Defocus  |
| Motion   |
| Zoom     |
| Elastic  |
| Gaussian |
| Impulse  |
| Shot     |
| Pixelate |
| JPEG     |
||
| Combo (W&L) |
| Combo (S&M) |
| Combo (D&P) |
||
| Combo (All) |

> **Model:** MonoDepth2, **Backbone:** ResNet-18, **Corrupt Probability:** 0.5, **Metric:** Abs Rel

| Train | Bright | Dark | Fog | Frost | Snow | Contrast | Defocus | Glass | Motion | Zoom | Elastic| Quant| Gaussian | Impulse | Shot | ISO | Pixelate | JPEG |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Bright   |.119|.202|.138|.202|.397|.160|.245|.225|.205|.167|.124|.156|.245|.255|.235|.246|.134|.159|
| Dark     |
| Fog      |
| Snow     |
| Contrast |
| Defocus  |
| Motion   |
| Zoom     |
| Elastic  |
| Gaussian |
| Impulse  |
| Shot     |
| Pixelate |
| JPEG     |
||
| Combo (W&L) |
| Combo (S&M) |
| Combo (D&P) |
||
| Combo (All) |

> **Model:** MonoDepth2, **Backbone:** ResNet-18, **Corrupt Probability:** 0.25, **Metric:** Abs Rel

| Train | Bright | Dark | Fog | Frost | Snow | Contrast | Defocus | Glass | Motion | Zoom | Elastic| Quant| Gaussian | Impulse | Shot | ISO | Pixelate | JPEG |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| Bright   | 
| Dark     |
| Fog      |
| Snow     |
| Contrast |
| Defocus  |
| Motion   |
| Zoom     |
| Elastic  |
| Gaussian |
| Impulse  |
| Shot     |
| Pixelate |
| JPEG     |
||
| Combo (W&L) |
| Combo (S&M) |
| Combo (D&P) |
||
| Combo (All) |

