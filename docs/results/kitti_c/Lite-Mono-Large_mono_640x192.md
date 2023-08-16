<img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/logo2.png" align="right" width="34%">

# RoboDepth Benchmark
The following metrics are consistently used in our benchmark:
- Absolute Relative Difference: $\text{Abs Rel} = \frac{1}{|D|}\sum_{pred\in D}\frac{|gt - pred|}{gt}$
- Accuracy: $\delta_t = \frac{1}{|D|}|\{\ pred\in D | \max{(\frac{gt}{pred}, \frac{pred}{gt})< 1.25^t}\}| \times 100\\%$
- Depth Estimation Score (DES):
  - $\text{DES}_1 = \text{Abs Rel} - \delta_1 + 1$ 
  - $\text{DES}_2 = \frac{\text{Abs Rel} - \delta_1 + 1}{2}$
  - $\text{DES}_3 = \frac{\text{Abs Rel}}{\delta_1}$


### Lite-Mono (Large), Mono, 640x192
| Corruption | Clean | Bright | Dark | Fog | Frost | Snow | Contrast | Defocus | Glass | Motion | Zoom | Elastic| Quant| Gaussian | Impulse | Shot | ISO | Pixelate | JPEG | 
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |          
| $\text{DEE}_{1}$ | 0.2040 | 0.2190     | 0.4540     | 0.2530     | 0.5100     | 0.8670     | 0.2990     | 0.4440     | 0.4490     | 0.4390     | 0.3850     | 0.2420     | 0.2960     | 0.7250     | 0.6960     | 0.6580     | 0.7240     | 0.3200     | 0.3680     |
| $\text{CE}_{1}$  |      -       | 0.8456  | 0.8093  | 0.8135  | 0.9222  | 0.8475  | 0.8016  | 0.9117  | 0.9277  | 1.0139  | 0.9577  | 0.9380  | 0.7668  | 0.9440  | 0.8935  | 0.9662  | 0.9330  | 1.1073  | 0.9412  |
| $\text{RCE}_{1}$ |      -       | 0.7143 | 0.7740 | 0.6712 | 0.9714 | 0.8446 | 0.7037 | 0.9639 | 0.9959 | 1.2051 | 1.1037 | 1.9000 | 0.6216 | 0.9830 | 0.9094 | 1.0248 | 0.9665 | 2.2745 | 1.0719 |
| $\text{RR}_{1}$  |      -       | 0.9812  | 0.6859  | 0.9384  | 0.6156  | 0.1671  | 0.8807  | 0.6985  | 0.6922  | 0.7048  | 0.7726  | 0.9523  | 0.8844  | 0.3455  | 0.3819  | 0.4296  | 0.3467  | 0.8543  | 0.7940  |
| $\text{DEE}_{2}$ | 0.1020 | 0.1100     | 0.2270     | 0.1260     | 0.2550     | 0.4330     | 0.1490     | 0.2220     | 0.2250     | 0.2200     | 0.1920     | 0.1210     | 0.1480     | 0.3630     | 0.3480     | 0.3290     | 0.3620     | 0.1600     | 0.1840     |
| $\text{CE}_{2}$  |      -       | 0.8462  | 0.8107  | 0.8129  | 0.9206  | 0.8474  | 0.7968  | 0.9098  | 0.9298  | 1.0185  | 0.9552  | 0.9380  | 0.7668  | 0.9453  | 0.8946  | 0.9676  | 0.9330  | 1.1034  | 0.9388  |
| $\text{RCE}_{2}$ |      -       | 0.7273 | 0.7764 | 0.6667 | 0.9684 | 0.8444 | 0.6912 | 0.9600 | 1.0000 | 1.2165 | 1.0976 | 1.9000 | 0.6216 | 0.9849 | 0.9111 | 1.0271 | 0.9665 | 2.2308 | 1.0649 |
| $\text{RR}_{2}$  |      -       | 0.9911  | 0.8608  | 0.9733  | 0.8296  | 0.6314  | 0.9477  | 0.8664  | 0.8630  | 0.8686  | 0.8998  | 0.9788  | 0.9488  | 0.7094  | 0.7261  | 0.7472  | 0.7105  | 0.9354  | 0.9087  |
| $\text{DEE}_{3}$ | 0.1130 | 0.1190     | 0.2470     | 0.1360     | 0.2860     | 0.7080     | 0.1580     | 0.2400     | 0.2440     | 0.2370     | 0.2090     | 0.1280     | 0.1570     | 0.4880     | 0.4530     | 0.4130     | 0.4830     | 0.1720     | 0.1960     |
| $\text{CE}_{3}$  |      -       | 0.8500  | 0.7485  | 0.8193  | 0.8746  | 0.6692  | 0.7940  | 0.8856  | 0.9037  | 1.0042  | 0.9457  | 0.9078  | 0.7512  | 0.8857  | 0.8046  | 0.9344  | 0.8671  | 1.0886  | 0.9116  |
| $\text{RCE}_{3}$ |      -       | 0.6667 | 0.6734 | 0.6571 | 0.8827 | 0.6419 | 0.6618 | 0.9071 | 0.9424 | 1.1810 | 1.0667 | 1.5000 | 0.5641 | 0.8929 | 0.7870 | 0.9646 | 0.8685 | 2.1852 | 0.9881 |
| $\text{RR}_{3}$  |      -       | 0.9932  | 0.8489  | 0.9741  | 0.8050  | 0.3292  | 0.9493  | 0.8568  | 0.8523  | 0.8602  | 0.8918  | 0.9831  | 0.9504  | 0.5772  | 0.6167  | 0.6618  | 0.5829  | 0.9335  | 0.9064  |

- **Summary:** $\text{mCE}_1 =$ 0.9078, $\text{RmCE}_1 =$ 1.0389, $\text{mRR}_1 =$ 0.6736, $\text{mCE}_2 =$ 0.9075, $\text{RmCE}_2 =$ 1.0364, $\text{mRR}_2 =$ 0.8554, $\text{mCE}_3 =$ 0.8692, $\text{RmCE}_3 =$ 0.9462, $\text{mRR}_3 =$ 0.8096


### Clean
| $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.101 | 0.732 | 4.459 | 0.178 | 0.897 | 0.965 | 0.983 |

- **Summary:** $\text{DEE}_1=$ 0.204, $\text{DEE}_2=$ 0.102, $\text{DEE}_3=$ 0.113


### Brightness
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.101 | 0.737 | 4.460 | 0.178 | 0.897 | 0.965 | 0.983 |
|   2   | 0.102 | 0.748 | 4.494 | 0.178 | 0.893 | 0.965 | 0.984 |
|   3   | 0.104 | 0.753 | 4.550 | 0.180 | 0.887 | 0.964 | 0.984 |
|   4   | 0.107 | 0.756 | 4.633 | 0.183 | 0.880 | 0.963 | 0.984 |
|   5   | 0.111 | 0.767 | 4.753 | 0.186 | 0.871 | 0.961 | 0.983 |
|  avg  | 0.105 | 0.752 | 4.578 | 0.181 | 0.886 | 0.964 | 0.984 |

- **Summary:** $\text{DEE}_1=$ 0.219, $\text{DEE}_2=$ 0.110, $\text{DEE}_3=$ 0.119


### Dark
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.127 | 0.918 | 5.128 | 0.205 | 0.844 | 0.951 | 0.980 |
|   2   | 0.144 | 1.027 | 5.481 | 0.221 | 0.806 | 0.941 | 0.978 |
|   3   | 0.169 | 1.249 | 6.112 | 0.249 | 0.750 | 0.917 | 0.970 |
|   4   | 0.204 | 1.614 | 7.235 | 0.294 | 0.667 | 0.878 | 0.953 |
|   5   | 0.251 | 2.181 | 8.651 | 0.358 | 0.556 | 0.826 | 0.922 |
|  avg  | 0.179 | 1.398 | 6.521 | 0.265 | 0.725 | 0.903 | 0.961 |

- **Summary:** $\text{DEE}_1=$ 0.454, $\text{DEE}_2=$ 0.227, $\text{DEE}_3=$ 0.247


### Fog
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.111 | 0.802 | 4.670 | 0.188 | 0.879 | 0.961 | 0.982 |
|   2   | 0.114 | 0.809 | 4.722 | 0.191 | 0.872 | 0.959 | 0.982 |
|   3   | 0.118 | 0.830 | 4.790 | 0.194 | 0.865 | 0.957 | 0.981 |
|   4   | 0.119 | 0.836 | 4.814 | 0.194 | 0.862 | 0.957 | 0.982 |
|   5   | 0.126 | 0.882 | 4.937 | 0.202 | 0.846 | 0.952 | 0.980 |
|  avg  | 0.118 | 0.832 | 4.787 | 0.194 | 0.865 | 0.957 | 0.981 |

- **Summary:** $\text{DEE}_1=$ 0.253, $\text{DEE}_2=$ 0.126, $\text{DEE}_3=$ 0.136


### Frost
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.130 | 0.927 | 4.976 | 0.205 | 0.836 | 0.951 | 0.980 |
|   2   | 0.175 | 1.280 | 5.803 | 0.252 | 0.733 | 0.918 | 0.969 |
|   3   | 0.211 | 1.587 | 6.450 | 0.289 | 0.648 | 0.885 | 0.957 |
|   4   | 0.219 | 1.674 | 6.616 | 0.297 | 0.636 | 0.875 | 0.954 |
|   5   | 0.246 | 1.956 | 7.114 | 0.324 | 0.577 | 0.848 | 0.943 |
|  avg  | 0.196 | 1.485 | 6.192 | 0.273 | 0.686 | 0.895 | 0.961 |

- **Summary:** $\text{DEE}_1=$ 0.510, $\text{DEE}_2=$ 0.255, $\text{DEE}_3=$ 0.286


### Snow
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.194 | 1.450 | 5.822 | 0.283 | 0.700 | 0.889 | 0.954 |
|   2   | 0.349 | 3.371 | 8.751 | 0.437 | 0.408 | 0.720 | 0.881 |
|   3   | 0.330 | 3.143 | 8.373 | 0.410 | 0.435 | 0.750 | 0.898 |
|   4   | 0.378 | 3.869 | 9.478 | 0.459 | 0.361 | 0.686 | 0.872 |
|   5   | 0.364 | 3.579 | 9.340 | 0.449 | 0.377 | 0.704 | 0.880 |
|  avg  | 0.323 | 3.082 | 8.353 | 0.408 | 0.456 | 0.750 | 0.897 |

- **Summary:** $\text{DEE}_1=$ 0.867, $\text{DEE}_2=$ 0.433, $\text{DEE}_3=$ 0.708


### Contrast
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.106 | 0.770 | 4.611 | 0.184 | 0.887 | 0.962 | 0.982 |
|   2   | 0.108 | 0.782 | 4.677 | 0.187 | 0.881 | 0.960 | 0.982 |
|   3   | 0.116 | 0.821 | 4.828 | 0.193 | 0.866 | 0.957 | 0.982 |
|   4   | 0.141 | 1.002 | 5.358 | 0.217 | 0.817 | 0.941 | 0.978 |
|   5   | 0.187 | 1.394 | 6.337 | 0.260 | 0.713 | 0.904 | 0.967 |
|  avg  | 0.132 | 0.954 | 5.162 | 0.208 | 0.833 | 0.945 | 0.978 |

- **Summary:** $\text{DEE}_1=$ 0.299, $\text{DEE}_2=$ 0.149, $\text{DEE}_3=$ 0.158


### Defocus Blur
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.124 | 0.947 | 5.524 | 0.212 | 0.842 | 0.944 | 0.976 |
|   2   | 0.143 | 1.147 | 6.209 | 0.239 | 0.803 | 0.926 | 0.968 |
|   3   | 0.181 | 1.541 | 7.246 | 0.287 | 0.722 | 0.893 | 0.950 |
|   4   | 0.207 | 1.809 | 7.879 | 0.319 | 0.663 | 0.868 | 0.938 |
|   5   | 0.224 | 1.998 | 8.319 | 0.339 | 0.627 | 0.849 | 0.928 |
|  avg  | 0.176 | 1.488 | 7.035 | 0.279 | 0.731 | 0.896 | 0.952 |

- **Summary:** $\text{DEE}_1=$ 0.444, $\text{DEE}_2=$ 0.222, $\text{DEE}_3=$ 0.240


### Glass Blur
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.124 | 0.999 | 5.547 | 0.206 | 0.847 | 0.946 | 0.978 |
|   2   | 0.138 | 1.153 | 6.177 | 0.228 | 0.814 | 0.930 | 0.970 |
|   3   | 0.187 | 1.690 | 7.801 | 0.298 | 0.711 | 0.879 | 0.944 |
|   4   | 0.206 | 1.905 | 8.352 | 0.325 | 0.669 | 0.860 | 0.931 |
|   5   | 0.235 | 2.222 | 8.954 | 0.362 | 0.604 | 0.831 | 0.915 |
|  avg  | 0.178 | 1.594 | 7.366 | 0.284 | 0.729 | 0.889 | 0.948 |

- **Summary:** $\text{DEE}_1=$ 0.449, $\text{DEE}_2=$ 0.225, $\text{DEE}_3=$ 0.244


### Motion Blur
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.116 | 0.858 | 5.006 | 0.196 | 0.865 | 0.955 | 0.981 |
|   2   | 0.135 | 1.027 | 5.722 | 0.221 | 0.821 | 0.937 | 0.975 |
|   3   | 0.172 | 1.393 | 6.796 | 0.269 | 0.737 | 0.904 | 0.959 |
|   4   | 0.213 | 1.811 | 7.699 | 0.316 | 0.648 | 0.863 | 0.941 |
|   5   | 0.234 | 2.030 | 8.108 | 0.342 | 0.602 | 0.840 | 0.928 |
|  avg  | 0.174 | 1.424 | 6.666 | 0.269 | 0.735 | 0.900 | 0.957 |

- **Summary:** $\text{DEE}_1=$ 0.439, $\text{DEE}_2=$ 0.220, $\text{DEE}_3=$ 0.237


### Zoom Blur
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.150 | 1.124 | 5.533 | 0.230 | 0.802 | 0.930 | 0.972 |
|   2   | 0.163 | 1.223 | 5.756 | 0.242 | 0.777 | 0.919 | 0.968 |
|   3   | 0.161 | 1.232 | 5.847 | 0.242 | 0.780 | 0.918 | 0.968 |
|   4   | 0.170 | 1.313 | 5.966 | 0.250 | 0.763 | 0.912 | 0.966 |
|   5   | 0.168 | 1.329 | 6.046 | 0.250 | 0.766 | 0.912 | 0.966 |
|  avg  | 0.162 | 1.244 | 5.830 | 0.243 | 0.778 | 0.918 | 0.968 |

- **Summary:** $\text{DEE}_1=$ 0.385, $\text{DEE}_2=$ 0.192, $\text{DEE}_3=$ 0.209


### Elastic Transform
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.104 | 0.742 | 4.593 | 0.182 | 0.888 | 0.962 | 0.983 |
|   2   | 0.105 | 0.749 | 4.695 | 0.185 | 0.883 | 0.960 | 0.982 |
|   3   | 0.110 | 0.786 | 4.909 | 0.192 | 0.872 | 0.956 | 0.981 |
|   4   | 0.115 | 0.823 | 5.097 | 0.198 | 0.861 | 0.952 | 0.980 |
|   5   | 0.123 | 0.895 | 5.371 | 0.208 | 0.845 | 0.945 | 0.978 |
|  avg  | 0.111 | 0.799 | 4.933 | 0.193 | 0.870 | 0.955 | 0.981 |

- **Summary:** $\text{DEE}_1=$ 0.242, $\text{DEE}_2=$ 0.121, $\text{DEE}_3=$ 0.128


### Color Quant
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.102 | 0.741 | 4.475 | 0.178 | 0.896 | 0.965 | 0.983 |
|   2   | 0.105 | 0.765 | 4.526 | 0.180 | 0.892 | 0.965 | 0.983 |
|   3   | 0.113 | 0.826 | 4.732 | 0.189 | 0.876 | 0.960 | 0.982 |
|   4   | 0.139 | 1.008 | 5.386 | 0.220 | 0.817 | 0.942 | 0.976 |
|   5   | 0.196 | 1.530 | 6.903 | 0.290 | 0.692 | 0.891 | 0.954 |
|  avg  | 0.131 | 0.974 | 5.204 | 0.211 | 0.835 | 0.945 | 0.976 |

- **Summary:** $\text{DEE}_1=$ 0.296, $\text{DEE}_2=$ 0.148, $\text{DEE}_3=$ 0.157


### Gaussian Noise
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.168 | 1.175 | 5.789 | 0.248 | 0.747 | 0.918 | 0.971 |
|   2   | 0.210 | 1.554 | 6.813 | 0.300 | 0.642 | 0.879 | 0.953 |
|   3   | 0.265 | 2.156 | 8.225 | 0.366 | 0.503 | 0.822 | 0.927 |
|   4   | 0.312 | 2.806 | 9.507 | 0.427 | 0.419 | 0.746 | 0.894 |
|   5   | 0.354 | 3.433 | 10.524 | 0.484 | 0.371 | 0.676 | 0.857 |
|  avg  | 0.262 | 2.225 | 8.172 | 0.365 | 0.536 | 0.808 | 0.920 |

- **Summary:** $\text{DEE}_1=$ 0.725, $\text{DEE}_2=$ 0.363, $\text{DEE}_3=$ 0.488


### Impulse Noise
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.146 | 0.993 | 5.403 | 0.226 | 0.797 | 0.938 | 0.977 |
|   2   | 0.200 | 1.444 | 6.707 | 0.288 | 0.663 | 0.887 | 0.958 |
|   3   | 0.246 | 1.933 | 7.830 | 0.344 | 0.548 | 0.838 | 0.933 |
|   4   | 0.316 | 2.875 | 9.695 | 0.435 | 0.412 | 0.739 | 0.887 |
|   5   | 0.354 | 3.475 | 10.649 | 0.490 | 0.364 | 0.673 | 0.854 |
|  avg  | 0.252 | 2.144 | 8.057 | 0.357 | 0.557 | 0.815 | 0.922 |

- **Summary:** $\text{DEE}_1=$ 0.696, $\text{DEE}_2=$ 0.348, $\text{DEE}_3=$ 0.453


### Shot Noise
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.132 | 0.892 | 5.205 | 0.205 | 0.829 | 0.951 | 0.982 |
|   2   | 0.173 | 1.247 | 6.429 | 0.256 | 0.726 | 0.912 | 0.970 |
|   3   | 0.236 | 1.916 | 8.091 | 0.335 | 0.571 | 0.848 | 0.936 |
|   4   | 0.316 | 2.972 | 10.010 | 0.440 | 0.417 | 0.737 | 0.882 |
|   5   | 0.349 | 3.466 | 10.742 | 0.486 | 0.375 | 0.683 | 0.854 |
|  avg  | 0.241 | 2.099 | 8.095 | 0.344 | 0.584 | 0.826 | 0.925 |

- **Summary:** $\text{DEE}_1=$ 0.658, $\text{DEE}_2=$ 0.329, $\text{DEE}_3=$ 0.413


### ISO Noise
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.197 | 1.473 | 6.953 | 0.286 | 0.667 | 0.889 | 0.959 |
|   2   | 0.219 | 1.693 | 7.451 | 0.313 | 0.615 | 0.868 | 0.947 |
|   3   | 0.253 | 2.063 | 8.232 | 0.354 | 0.529 | 0.834 | 0.930 |
|   4   | 0.291 | 2.533 | 9.128 | 0.401 | 0.452 | 0.782 | 0.907 |
|   5   | 0.326 | 3.062 | 10.030 | 0.450 | 0.402 | 0.720 | 0.879 |
|  avg  | 0.257 | 2.165 | 8.359 | 0.361 | 0.533 | 0.819 | 0.924 |

- **Summary:** $\text{DEE}_1=$ 0.724, $\text{DEE}_2=$ 0.362, $\text{DEE}_3=$ 0.483


### Pixelate
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.118 | 0.887 | 4.760 | 0.192 | 0.872 | 0.959 | 0.982 |
|   2   | 0.125 | 0.961 | 4.932 | 0.198 | 0.862 | 0.956 | 0.981 |
|   3   | 0.134 | 1.037 | 5.180 | 0.207 | 0.843 | 0.950 | 0.979 |
|   4   | 0.154 | 1.143 | 5.733 | 0.232 | 0.791 | 0.932 | 0.973 |
|   5   | 0.175 | 1.352 | 6.560 | 0.265 | 0.740 | 0.909 | 0.961 |
|  avg  | 0.141 | 1.076 | 5.433 | 0.219 | 0.822 | 0.941 | 0.975 |

- **Summary:** $\text{DEE}_1=$ 0.320, $\text{DEE}_2=$ 0.160, $\text{DEE}_3=$ 0.172


### JPEG Compression
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.121 | 0.875 | 5.028 | 0.200 | 0.857 | 0.954 | 0.980 |
|   2   | 0.131 | 0.940 | 5.242 | 0.210 | 0.837 | 0.949 | 0.978 |
|   3   | 0.140 | 0.999 | 5.423 | 0.220 | 0.818 | 0.942 | 0.976 |
|   4   | 0.170 | 1.252 | 6.129 | 0.255 | 0.750 | 0.918 | 0.967 |
|   5   | 0.208 | 1.665 | 7.163 | 0.303 | 0.667 | 0.881 | 0.948 |
|  avg  | 0.154 | 1.146 | 5.797 | 0.238 | 0.786 | 0.929 | 0.970 |

- **Summary:** $\text{DEE}_1=$ 0.368, $\text{DEE}_2=$ 0.184, $\text{DEE}_3=$ 0.196
