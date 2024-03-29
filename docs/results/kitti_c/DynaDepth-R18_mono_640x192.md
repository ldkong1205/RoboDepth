<img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/logo2.png" align="right" width="34%">

# RoboDepth Benchmark
The following metrics are consistently used in our benchmark:
- Absolute Relative Difference: $\text{Abs Rel} = \frac{1}{|D|}\sum_{pred\in D}\frac{|gt - pred|}{gt}$
- Accuracy: $\delta_t = \frac{1}{|D|}|\{\ pred\in D | \max{(\frac{gt}{pred}, \frac{pred}{gt})< 1.25^t}\}| \times 100\\%$
- Depth Estimation Score (DES):
  - $\text{DES}_1 = \text{Abs Rel} - \delta_1 + 1$ 
  - $\text{DES}_2 = \frac{\text{Abs Rel} - \delta_1 + 1}{2}$
  - $\text{DES}_3 = \frac{\text{Abs Rel}}{\delta_1}$


### DynaDepth (R18), Mono, 640x192
| Corruption | Clean | Bright | Dark | Fog | Frost | Snow | Contrast | Defocus | Glass | Motion | Zoom | Elastic| Quant| Gaussian | Impulse | Shot | ISO | Pixelate | JPEG | 
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |          
| $\text{DEE}_{1}$ | 0.2350 | 0.2570     | 0.5770     | 0.3130     | 0.5790     | 1.0180     | 0.4150     | 1.0010     | 0.6940     | 0.6090     | 0.4150     | 0.2550     | 0.3730     | 0.7570     | 0.7570     | 0.6720     | 0.7590     | 0.2820     | 0.3590     |
| $\text{CE}_{1}$  |      -       | 0.9923  | 1.0285  | 1.0064  | 1.0470  | 0.9951  | 1.1126  | 2.0554  | 1.4339  | 1.4065  | 1.0323  | 0.9884  | 0.9663  | 0.9857  | 0.9718  | 0.9868  | 0.9781  | 0.9758  | 0.9182  |
| $\text{RCE}_{1}$ |      -       | 1.0476 | 1.0588 | 1.0685 | 1.0921 | 0.9975 | 1.3333 | 3.0763 | 1.8659 | 1.9179 | 1.0976 | 1.0000 | 0.9324 | 0.9849 | 0.9649 | 0.9865 | 0.9740 | 0.9216 | 0.8105 |
| $\text{RR}_{1}$  |      -       | 0.9712  | 0.5529  | 0.8980  | 0.5503  | -0.0235  | 0.7647  | -0.0013  | 0.4000  | 0.5111  | 0.7647  | 0.9739  | 0.8196  | 0.3176  | 0.3176  | 0.4288  | 0.3150  | 0.9386  | 0.8379  |
| $\text{DEE}_{2}$ | 0.1170 | 0.1280     | 0.2890     | 0.1560     | 0.2890     | 0.5090     | 0.2080     | 0.5010     | 0.3470     | 0.3050     | 0.2070     | 0.1270     | 0.1860     | 0.3790     | 0.3790     | 0.3360     | 0.3790     | 0.1410     | 0.1800     |
| $\text{CE}_{2}$  |      -       | 0.9846  | 1.0321  | 1.0065  | 1.0433  | 0.9961  | 1.1123  | 2.0533  | 1.4339  | 1.4120  | 1.0299  | 0.9845  | 0.9637  | 0.9870  | 0.9743  | 0.9882  | 0.9768  | 0.9724  | 0.9184  |
| $\text{RCE}_{2}$ |      -       | 1.0000 | 1.0683 | 1.0833 | 1.0886 | 1.0000 | 1.3382 | 3.0720 | 1.8699 | 1.9381 | 1.0976 | 1.0000 | 0.9324 | 0.9887 | 0.9704 | 0.9910 | 0.9740 | 0.9231 | 0.8182 |
| $\text{RR}_{2}$  |      -       | 0.9875  | 0.8052  | 0.9558  | 0.8052  | 0.5561  | 0.8969  | 0.5651  | 0.7395  | 0.7871  | 0.8981  | 0.9887  | 0.9219  | 0.7033  | 0.7033  | 0.7520  | 0.7033  | 0.9728  | 0.9287  |
| $\text{DEE}_{3}$ | 0.1280 | 0.1370     | 0.3470     | 0.1650     | 0.3490     | 1.0460     | 0.2250     | 1.0030     | 0.4690     | 0.3870     | 0.2270     | 0.1370     | 0.1980     | 0.5380     | 0.5330     | 0.4350     | 0.5350     | 0.1530     | 0.1950     |
| $\text{CE}_{3}$  |      -       | 0.9786  | 1.0515  | 0.9940  | 1.0673  | 0.9887  | 1.1307  | 3.7011  | 1.7370  | 1.6398  | 1.0271  | 0.9716  | 0.9474  | 0.9764  | 0.9467  | 0.9842  | 0.9605  | 0.9684  | 0.9070  |
| $\text{RCE}_{3}$ |      -       | 1.0000 | 1.1005 | 1.0571 | 1.1276 | 0.9903 | 1.4265 | 6.2500 | 2.4532 | 2.4667 | 1.1000 | 0.9000 | 0.8974 | 0.9762 | 0.9375 | 0.9871 | 0.9554 | 0.9259 | 0.7976 |
| $\text{RR}_{3}$  |      -       | 0.9897  | 0.7489  | 0.9576  | 0.7466  | -0.0528  | 0.8888  | -0.0034  | 0.6089  | 0.7030  | 0.8865  | 0.9897  | 0.9197  | 0.5298  | 0.5356  | 0.6479  | 0.5333  | 0.9713  | 0.9232  |

- **Summary:** $\text{mCE}_1 =$ 1.1045, $\text{RmCE}_1 =$ 1.2295, $\text{mRR}_1 =$ 0.5743, $\text{mCE}_2 =$ 1.1038, $\text{RmCE}_2 =$ 1.2308, $\text{mRR}_2 =$ 0.8150, $\text{mCE}_3 =$ 1.2210, $\text{RmCE}_3 =$ 1.4638, $\text{mRR}_3 =$ 0.6958


### Clean
| $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.112 | 0.839 | 4.848 | 0.192 | 0.877 | 0.959 | 0.981 |

- **Summary:** $\text{DEE}_1=$ 0.235, $\text{DEE}_2=$ 0.117, $\text{DEE}_3=$ 0.128


### Brightness
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.112 | 0.839 | 4.852 | 0.192 | 0.876 | 0.959 | 0.981 |
|   2   | 0.114 | 0.858 | 4.914 | 0.195 | 0.870 | 0.958 | 0.981 |
|   3   | 0.117 | 0.881 | 4.997 | 0.198 | 0.863 | 0.956 | 0.981 |
|   4   | 0.121 | 0.910 | 5.112 | 0.203 | 0.854 | 0.953 | 0.980 |
|   5   | 0.125 | 0.936 | 5.263 | 0.209 | 0.842 | 0.948 | 0.979 |
|  avg  | 0.118 | 0.885 | 5.028 | 0.199 | 0.861 | 0.955 | 0.980 |

- **Summary:** $\text{DEE}_1=$ 0.257, $\text{DEE}_2=$ 0.128, $\text{DEE}_3=$ 0.137


### Dark
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.143 | 1.115 | 5.635 | 0.227 | 0.814 | 0.936 | 0.974 |
|   2   | 0.156 | 1.207 | 5.940 | 0.243 | 0.782 | 0.925 | 0.969 |
|   3   | 0.192 | 1.508 | 6.600 | 0.286 | 0.698 | 0.890 | 0.953 |
|   4   | 0.275 | 2.371 | 7.860 | 0.369 | 0.535 | 0.804 | 0.916 |
|   5   | 0.358 | 3.356 | 9.183 | 0.442 | 0.409 | 0.708 | 0.873 |
|  avg  | 0.225 | 1.911 | 7.044 | 0.313 | 0.648 | 0.853 | 0.937 |

- **Summary:** $\text{DEE}_1=$ 0.577, $\text{DEE}_2=$ 0.289, $\text{DEE}_3=$ 0.347


### Fog
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.126 | 0.965 | 5.209 | 0.209 | 0.843 | 0.948 | 0.978 |
|   2   | 0.131 | 1.011 | 5.334 | 0.216 | 0.836 | 0.944 | 0.976 |
|   3   | 0.137 | 1.060 | 5.483 | 0.224 | 0.820 | 0.939 | 0.974 |
|   4   | 0.137 | 1.056 | 5.517 | 0.225 | 0.821 | 0.938 | 0.974 |
|   5   | 0.149 | 1.160 | 5.780 | 0.240 | 0.796 | 0.928 | 0.970 |
|  avg  | 0.136 | 1.050 | 5.465 | 0.223 | 0.823 | 0.939 | 0.974 |

- **Summary:** $\text{DEE}_1=$ 0.313, $\text{DEE}_2=$ 0.156, $\text{DEE}_3=$ 0.165


### Frost
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.142 | 1.109 | 5.454 | 0.226 | 0.811 | 0.935 | 0.973 |
|   2   | 0.194 | 1.609 | 6.379 | 0.289 | 0.704 | 0.882 | 0.947 |
|   3   | 0.241 | 2.175 | 7.253 | 0.343 | 0.611 | 0.839 | 0.924 |
|   4   | 0.254 | 2.286 | 7.451 | 0.360 | 0.587 | 0.820 | 0.916 |
|   5   | 0.296 | 2.818 | 8.137 | 0.402 | 0.519 | 0.774 | 0.893 |
|  avg  | 0.225 | 1.999 | 6.935 | 0.324 | 0.646 | 0.850 | 0.931 |

- **Summary:** $\text{DEE}_1=$ 0.579, $\text{DEE}_2=$ 0.289, $\text{DEE}_3=$ 0.349


### Snow
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.249 | 2.245 | 7.048 | 0.350 | 0.600 | 0.828 | 0.917 |
|   2   | 0.413 | 4.645 | 9.973 | 0.502 | 0.354 | 0.642 | 0.825 |
|   3   | 0.418 | 4.994 | 10.141 | 0.508 | 0.359 | 0.640 | 0.821 |
|   4   | 0.477 | 6.172 | 11.354 | 0.554 | 0.302 | 0.576 | 0.783 |
|   5   | 0.459 | 5.566 | 10.902 | 0.547 | 0.312 | 0.595 | 0.789 |
|  avg  | 0.403 | 4.724 | 9.884 | 0.492 | 0.385 | 0.656 | 0.827 |

- **Summary:** $\text{DEE}_1=$ 1.018, $\text{DEE}_2=$ 0.509, $\text{DEE}_3=$ 1.046


### Contrast
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.120 | 0.926 | 5.102 | 0.202 | 0.859 | 0.953 | 0.980 |
|   2   | 0.126 | 0.969 | 5.240 | 0.209 | 0.847 | 0.950 | 0.978 |
|   3   | 0.138 | 1.066 | 5.523 | 0.225 | 0.819 | 0.939 | 0.975 |
|   4   | 0.179 | 1.402 | 6.424 | 0.280 | 0.733 | 0.902 | 0.957 |
|   5   | 0.284 | 2.561 | 8.715 | 0.421 | 0.512 | 0.785 | 0.892 |
|  avg  | 0.169 | 1.385 | 6.201 | 0.267 | 0.754 | 0.906 | 0.956 |

- **Summary:** $\text{DEE}_1=$ 0.415, $\text{DEE}_2=$ 0.208, $\text{DEE}_3=$ 0.225


### Defocus Blur
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.184 | 1.528 | 6.663 | 0.290 | 0.722 | 0.891 | 0.947 |
|   2   | 0.310 | 3.113 | 8.931 | 0.429 | 0.515 | 0.750 | 0.872 |
|   3   | 0.505 | 6.606 | 12.188 | 0.610 | 0.304 | 0.556 | 0.737 |
|   4   | 0.553 | 7.496 | 12.878 | 0.651 | 0.272 | 0.514 | 0.701 |
|   5   | 0.543 | 7.172 | 12.746 | 0.649 | 0.276 | 0.518 | 0.706 |
|  avg  | 0.419 | 5.183 | 10.681 | 0.526 | 0.418 | 0.646 | 0.793 |

- **Summary:** $\text{DEE}_1=$ 1.001, $\text{DEE}_2=$ 0.501, $\text{DEE}_3=$ 1.003


### Glass Blur
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.138 | 1.139 | 5.814 | 0.227 | 0.820 | 0.937 | 0.972 |
|   2   | 0.168 | 1.402 | 6.705 | 0.271 | 0.752 | 0.905 | 0.956 |
|   3   | 0.255 | 2.278 | 8.474 | 0.372 | 0.567 | 0.815 | 0.913 |
|   4   | 0.323 | 3.069 | 9.604 | 0.445 | 0.444 | 0.737 | 0.872 |
|   5   | 0.466 | 5.166 | 11.706 | 0.589 | 0.295 | 0.563 | 0.761 |
|  avg  | 0.270 | 2.611 | 8.461 | 0.381 | 0.576 | 0.791 | 0.895 |

- **Summary:** $\text{DEE}_1=$ 0.694, $\text{DEE}_2=$ 0.347, $\text{DEE}_3=$ 0.469


### Motion Blur
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.130 | 1.053 | 5.446 | 0.215 | 0.841 | 0.945 | 0.976 |
|   2   | 0.153 | 1.245 | 6.132 | 0.250 | 0.787 | 0.921 | 0.964 |
|   3   | 0.228 | 2.138 | 7.700 | 0.342 | 0.641 | 0.836 | 0.922 |
|   4   | 0.332 | 3.581 | 9.486 | 0.450 | 0.490 | 0.725 | 0.853 |
|   5   | 0.388 | 4.520 | 10.386 | 0.504 | 0.425 | 0.669 | 0.816 |
|  avg  | 0.246 | 2.507 | 7.830 | 0.352 | 0.637 | 0.819 | 0.906 |

- **Summary:** $\text{DEE}_1=$ 0.609, $\text{DEE}_2=$ 0.305, $\text{DEE}_3=$ 0.387


### Zoom Blur
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.162 | 1.394 | 6.021 | 0.251 | 0.782 | 0.923 | 0.965 |
|   2   | 0.184 | 1.564 | 6.365 | 0.279 | 0.733 | 0.900 | 0.954 |
|   3   | 0.164 | 1.374 | 6.144 | 0.253 | 0.772 | 0.916 | 0.965 |
|   4   | 0.177 | 1.469 | 6.347 | 0.267 | 0.745 | 0.904 | 0.959 |
|   5   | 0.172 | 1.417 | 6.279 | 0.260 | 0.754 | 0.910 | 0.962 |
|  avg  | 0.172 | 1.444 | 6.231 | 0.262 | 0.757 | 0.911 | 0.961 |

- **Summary:** $\text{DEE}_1=$ 0.415, $\text{DEE}_2=$ 0.207, $\text{DEE}_3=$ 0.227


### Elastic Transform
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.115 | 0.906 | 4.985 | 0.196 | 0.871 | 0.957 | 0.981 |
|   2   | 0.116 | 0.913 | 5.028 | 0.197 | 0.870 | 0.956 | 0.980 |
|   3   | 0.118 | 0.933 | 5.124 | 0.201 | 0.865 | 0.955 | 0.980 |
|   4   | 0.120 | 0.942 | 5.180 | 0.202 | 0.860 | 0.953 | 0.979 |
|   5   | 0.123 | 0.979 | 5.315 | 0.207 | 0.852 | 0.950 | 0.978 |
|  avg  | 0.118 | 0.935 | 5.126 | 0.201 | 0.864 | 0.954 | 0.980 |

- **Summary:** $\text{DEE}_1=$ 0.255, $\text{DEE}_2=$ 0.127, $\text{DEE}_3=$ 0.137


### Color Quant
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.112 | 0.840 | 4.857 | 0.192 | 0.876 | 0.959 | 0.981 |
|   2   | 0.113 | 0.858 | 4.907 | 0.194 | 0.873 | 0.959 | 0.981 |
|   3   | 0.124 | 0.937 | 5.099 | 0.206 | 0.852 | 0.952 | 0.979 |
|   4   | 0.167 | 1.288 | 5.984 | 0.252 | 0.759 | 0.921 | 0.968 |
|   5   | 0.259 | 2.182 | 7.891 | 0.353 | 0.551 | 0.840 | 0.932 |
|  avg  | 0.155 | 1.221 | 5.748 | 0.239 | 0.782 | 0.926 | 0.968 |

- **Summary:** $\text{DEE}_1=$ 0.373, $\text{DEE}_2=$ 0.186, $\text{DEE}_3=$ 0.198


### Gaussian Noise
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.165 | 1.246 | 5.981 | 0.253 | 0.752 | 0.912 | 0.965 |
|   2   | 0.217 | 1.722 | 7.027 | 0.314 | 0.638 | 0.860 | 0.941 |
|   3   | 0.292 | 2.572 | 8.484 | 0.392 | 0.488 | 0.779 | 0.904 |
|   4   | 0.352 | 3.231 | 9.460 | 0.446 | 0.396 | 0.704 | 0.872 |
|   5   | 0.390 | 3.673 | 10.160 | 0.483 | 0.356 | 0.648 | 0.846 |
|  avg  | 0.283 | 2.489 | 8.222 | 0.378 | 0.526 | 0.781 | 0.906 |

- **Summary:** $\text{DEE}_1=$ 0.757, $\text{DEE}_2=$ 0.379, $\text{DEE}_3=$ 0.538


### Impulse Noise
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.188 | 1.470 | 6.826 | 0.285 | 0.697 | 0.888 | 0.955 |
|   2   | 0.235 | 1.973 | 7.996 | 0.343 | 0.591 | 0.836 | 0.928 |
|   3   | 0.270 | 2.403 | 8.802 | 0.384 | 0.516 | 0.796 | 0.907 |
|   4   | 0.328 | 3.050 | 9.740 | 0.439 | 0.416 | 0.726 | 0.877 |
|   5   | 0.363 | 3.384 | 10.109 | 0.466 | 0.377 | 0.680 | 0.860 |
|  avg  | 0.277 | 2.456 | 8.695 | 0.383 | 0.519 | 0.785 | 0.905 |

- **Summary:** $\text{DEE}_1=$ 0.757, $\text{DEE}_2=$ 0.379, $\text{DEE}_3=$ 0.533


### Shot Noise
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.131 | 0.953 | 5.402 | 0.213 | 0.829 | 0.946 | 0.978 |
|   2   | 0.169 | 1.260 | 6.387 | 0.258 | 0.736 | 0.908 | 0.966 |
|   3   | 0.247 | 2.030 | 7.981 | 0.345 | 0.562 | 0.822 | 0.930 |
|   4   | 0.345 | 3.039 | 9.195 | 0.430 | 0.404 | 0.712 | 0.882 |
|   5   | 0.373 | 3.366 | 9.761 | 0.461 | 0.375 | 0.671 | 0.861 |
|  avg  | 0.253 | 2.130 | 7.745 | 0.341 | 0.581 | 0.812 | 0.923 |

- **Summary:** $\text{DEE}_1=$ 0.672, $\text{DEE}_2=$ 0.336, $\text{DEE}_3=$ 0.435


### ISO Noise
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.199 | 1.534 | 6.946 | 0.294 | 0.668 | 0.877 | 0.952 |
|   2   | 0.227 | 1.824 | 7.484 | 0.326 | 0.607 | 0.845 | 0.936 |
|   3   | 0.275 | 2.338 | 8.291 | 0.374 | 0.512 | 0.796 | 0.913 |
|   4   | 0.325 | 2.902 | 9.062 | 0.420 | 0.428 | 0.738 | 0.889 |
|   5   | 0.364 | 3.343 | 9.668 | 0.456 | 0.382 | 0.686 | 0.866 |
|  avg  | 0.278 | 2.388 | 8.290 | 0.374 | 0.519 | 0.788 | 0.911 |

- **Summary:** $\text{DEE}_1=$ 0.759, $\text{DEE}_2=$ 0.379, $\text{DEE}_3=$ 0.535


### Pixelate
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.116 | 0.926 | 4.994 | 0.196 | 0.871 | 0.958 | 0.981 |
|   2   | 0.118 | 0.947 | 5.052 | 0.198 | 0.867 | 0.956 | 0.980 |
|   3   | 0.125 | 1.023 | 5.234 | 0.205 | 0.856 | 0.952 | 0.979 |
|   4   | 0.141 | 1.214 | 5.634 | 0.222 | 0.829 | 0.941 | 0.974 |
|   5   | 0.148 | 1.327 | 5.851 | 0.232 | 0.817 | 0.935 | 0.971 |
|  avg  | 0.130 | 1.087 | 5.353 | 0.211 | 0.848 | 0.948 | 0.977 |

- **Summary:** $\text{DEE}_1=$ 0.282, $\text{DEE}_2=$ 0.141, $\text{DEE}_3=$ 0.153


### JPEG Compression
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.127 | 1.035 | 5.274 | 0.208 | 0.850 | 0.951 | 0.978 |
|   2   | 0.135 | 1.137 | 5.475 | 0.217 | 0.832 | 0.945 | 0.976 |
|   3   | 0.143 | 1.218 | 5.627 | 0.225 | 0.818 | 0.940 | 0.974 |
|   4   | 0.169 | 1.477 | 6.042 | 0.250 | 0.767 | 0.923 | 0.967 |
|   5   | 0.202 | 1.900 | 6.597 | 0.282 | 0.713 | 0.895 | 0.955 |
|  avg  | 0.155 | 1.353 | 5.803 | 0.236 | 0.796 | 0.931 | 0.970 |

- **Summary:** $\text{DEE}_1=$ 0.359, $\text{DEE}_2=$ 0.180, $\text{DEE}_3=$ 0.195

