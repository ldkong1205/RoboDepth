<img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/logo2.png" align="right" width="34%">

# RoboDepth Benchmark
The following metrics are consistently used in our benchmark:
- Absolute Relative Difference: $\text{Abs Rel} = \frac{1}{|D|}\sum_{pred\in D}\frac{|gt - pred|}{gt}$
- Accuracy: $\delta_t = \frac{1}{|D|}|\{\ pred\in D | \max{(\frac{gt}{pred}, \frac{pred}{gt})< 1.25^t}\}| \times 100\\%$
- Depth Estimation Score (DES):
  - $\text{DES}_1 = \text{Abs Rel} - \delta_1 + 1$ 
  - $\text{DES}_2 = \frac{\text{Abs Rel} - \delta_1 + 1}{2}$
  - $\text{DES}_3 = \frac{\text{Abs Rel}}{\delta_1}$


### Lite-Mono (Small), Mono, 640x192
| Corruption | Clean | Bright | Dark | Fog | Frost | Snow | Contrast | Defocus | Glass | Motion | Zoom | Elastic| Quant| Gaussian | Impulse | Shot | ISO | Pixelate | JPEG | 
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |          
| $\text{DEE}_{1}$ | 0.2310 | 0.2550     | 0.5030     | 0.3240     | 0.5030     | 0.8600     | 0.4750     | 0.7060     | 0.5640     | 0.4920     | 0.4070     | 0.2550     | 0.3220     | 0.7000     | 0.6730     | 0.6380     | 0.7130     | 0.3070     | 0.3280     |
| $\text{CE}_{1}$  |      -       | 0.9846  | 0.8966  | 1.0418  | 0.9096  | 0.8407  | 1.2735  | 1.4497  | 1.1653  | 1.1363  | 1.0124  | 0.9884  | 0.8342  | 0.9115  | 0.8639  | 0.9369  | 0.9188  | 1.0623  | 0.8389  |
| $\text{RCE}_{1}$ |      -       | 1.1429 | 0.8421 | 1.2740 | 0.8635 | 0.8013 | 1.8074 | 1.9076 | 1.3537 | 1.3385 | 1.0732 | 1.2000 | 0.6149 | 0.8849 | 0.8170 | 0.9187 | 0.8959 | 1.4902 | 0.6340 |
| $\text{RR}_{1}$  |      -       | 0.9688  | 0.6463  | 0.8791  | 0.6463  | 0.1821  | 0.6827  | 0.3823  | 0.5670  | 0.6606  | 0.7711  | 0.9688  | 0.8817  | 0.3901  | 0.4252  | 0.4707  | 0.3732  | 0.9012  | 0.8739  |
| $\text{DEE}_{2}$ | 0.1150 | 0.1270     | 0.2510     | 0.1620     | 0.2510     | 0.4300     | 0.2380     | 0.3530     | 0.2820     | 0.2460     | 0.2040     | 0.1280     | 0.1610     | 0.3500     | 0.3360     | 0.3190     | 0.3560     | 0.1540     | 0.1640     |
| $\text{CE}_{2}$  |      -       | 0.9769  | 0.8964  | 1.0452  | 0.9061  | 0.8415  | 1.2727  | 1.4467  | 1.1653  | 1.1389  | 1.0149  | 0.9922  | 0.8342  | 0.9115  | 0.8638  | 0.9382  | 0.9175  | 1.0621  | 0.8367  |
| $\text{RCE}_{2}$ |      -       | 1.0909 | 0.8447 | 1.3056 | 0.8608 | 0.8036 | 1.8088 | 1.9040 | 1.3577 | 1.3505 | 1.0854 | 1.3000 | 0.6216 | 0.8868 | 0.8185 | 0.9231 | 0.8959 | 1.5000 | 0.6364 |
| $\text{RR}_{2}$  |      -       | 0.9864  | 0.8463  | 0.9469  | 0.8463  | 0.6441  | 0.8610  | 0.7311  | 0.8113  | 0.8520  | 0.8994  | 0.9853  | 0.9480  | 0.7345  | 0.7503  | 0.7695  | 0.7277  | 0.9559  | 0.9446  |
| $\text{DEE}_{3}$ | 0.1250 | 0.1360     | 0.2820     | 0.1730     | 0.2810     | 0.6960     | 0.2630     | 0.4730     | 0.3280     | 0.2720     | 0.2200     | 0.1360     | 0.1700     | 0.4580     | 0.4260     | 0.3910     | 0.4700     | 0.1680     | 0.1750     |
| $\text{CE}_{3}$  |      -       | 0.9714  | 0.8545  | 1.0422  | 0.8593  | 0.6578  | 1.3216  | 1.7454  | 1.2148  | 1.1525  | 0.9955  | 0.9645  | 0.8134  | 0.8312  | 0.7567  | 0.8846  | 0.8438  | 1.0633  | 0.8140  |
| $\text{RCE}_{3}$ |      -       | 1.2222 | 0.7889 | 1.3714 | 0.7959 | 0.6160 | 2.0294 | 2.4857 | 1.4604 | 1.4000 | 1.0556 | 1.1000 | 0.5769 | 0.7929 | 0.6968 | 0.8553 | 0.8099 | 1.5926 | 0.5952 |
| $\text{RR}_{3}$  |      -       | 0.9874  | 0.8206  | 0.9451  | 0.8217  | 0.3474  | 0.8423  | 0.6023  | 0.7680  | 0.8320  | 0.8914  | 0.9874  | 0.9486  | 0.6194  | 0.6560  | 0.6960  | 0.6057  | 0.9509  | 0.9429  |

- **Summary:** $\text{mCE}_1 =$ 1.0036, $\text{RmCE}_1 =$ 1.1033, $\text{mRR}_1 =$ 0.6484, $\text{mCE}_2 =$ 1.0034, $\text{RmCE}_2 =$ 1.1108, $\text{mRR}_2 =$ 0.8467, $\text{mCE}_3 =$ 0.9881, $\text{RmCE}_3 =$ 1.1247, $\text{mRR}_3 =$ 0.7925


### Clean
| $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 0.110 | 0.809 | 4.677 | 0.186 | 0.879 | 0.961 | 0.982 |

- **Summary:** $\text{DEE}_1=$ 0.231, $\text{DEE}_2=$ 0.115, $\text{DEE}_3=$ 0.125


### Brightness
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.110 | 0.807 | 4.671 | 0.186 | 0.879 | 0.961 | 0.982 |
|   2   | 0.112 | 0.816 | 4.702 | 0.187 | 0.874 | 0.961 | 0.983 |
|   3   | 0.116 | 0.830 | 4.761 | 0.190 | 0.866 | 0.960 | 0.983 |
|   4   | 0.122 | 0.848 | 4.840 | 0.194 | 0.855 | 0.957 | 0.983 |
|   5   | 0.128 | 0.871 | 4.952 | 0.200 | 0.841 | 0.954 | 0.982 |
|  avg  | 0.118 | 0.834 | 4.785 | 0.191 | 0.863 | 0.959 | 0.983 |

- **Summary:** $\text{DEE}_1=$ 0.255, $\text{DEE}_2=$ 0.127, $\text{DEE}_3=$ 0.136


### Dark
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.141 | 1.093 | 5.462 | 0.219 | 0.820 | 0.941 | 0.977 |
|   2   | 0.154 | 1.177 | 5.749 | 0.232 | 0.791 | 0.932 | 0.975 |
|   3   | 0.180 | 1.369 | 6.259 | 0.259 | 0.731 | 0.911 | 0.968 |
|   4   | 0.220 | 1.754 | 7.414 | 0.308 | 0.632 | 0.869 | 0.948 |
|   5   | 0.282 | 2.502 | 9.181 | 0.391 | 0.489 | 0.796 | 0.907 |
|  avg  | 0.195 | 1.579 | 6.813 | 0.282 | 0.693 | 0.890 | 0.955 |

- **Summary:** $\text{DEE}_1=$ 0.503, $\text{DEE}_2=$ 0.251, $\text{DEE}_3=$ 0.282


### Fog
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.129 | 0.966 | 5.060 | 0.206 | 0.841 | 0.949 | 0.979 |
|   2   | 0.136 | 1.019 | 5.188 | 0.213 | 0.830 | 0.945 | 0.978 |
|   3   | 0.145 | 1.079 | 5.336 | 0.221 | 0.814 | 0.939 | 0.976 |
|   4   | 0.144 | 1.058 | 5.350 | 0.221 | 0.813 | 0.939 | 0.977 |
|   5   | 0.154 | 1.145 | 5.563 | 0.232 | 0.792 | 0.931 | 0.973 |
|  avg  | 0.142 | 1.053 | 5.299 | 0.219 | 0.818 | 0.941 | 0.977 |

- **Summary:** $\text{DEE}_1=$ 0.324, $\text{DEE}_2=$ 0.162, $\text{DEE}_3=$ 0.173


### Frost
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.141 | 1.032 | 5.221 | 0.216 | 0.814 | 0.943 | 0.978 |
|   2   | 0.177 | 1.325 | 5.925 | 0.255 | 0.731 | 0.914 | 0.968 |
|   3   | 0.206 | 1.587 | 6.499 | 0.286 | 0.658 | 0.889 | 0.958 |
|   4   | 0.213 | 1.658 | 6.605 | 0.291 | 0.649 | 0.882 | 0.956 |
|   5   | 0.234 | 1.844 | 6.941 | 0.312 | 0.604 | 0.864 | 0.948 |
|  avg  | 0.194 | 1.489 | 6.238 | 0.272 | 0.691 | 0.898 | 0.962 |

- **Summary:** $\text{DEE}_1=$ 0.503, $\text{DEE}_2=$ 0.251, $\text{DEE}_3=$ 0.281


### Snow
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.197 | 1.519 | 5.951 | 0.291 | 0.693 | 0.883 | 0.947 |
|   2   | 0.346 | 3.275 | 8.546 | 0.435 | 0.403 | 0.724 | 0.889 |
|   3   | 0.322 | 2.985 | 8.046 | 0.416 | 0.445 | 0.751 | 0.892 |
|   4   | 0.368 | 3.787 | 9.020 | 0.452 | 0.382 | 0.695 | 0.872 |
|   5   | 0.364 | 3.500 | 8.992 | 0.450 | 0.372 | 0.695 | 0.878 |
|  avg  | 0.319 | 3.013 | 8.111 | 0.409 | 0.459 | 0.750 | 0.896 |

- **Summary:** $\text{DEE}_1=$ 0.860, $\text{DEE}_2=$ 0.430, $\text{DEE}_3=$ 0.696


### Contrast
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.120 | 0.889 | 4.933 | 0.197 | 0.858 | 0.954 | 0.980 |
|   2   | 0.128 | 0.943 | 5.101 | 0.207 | 0.840 | 0.949 | 0.979 |
|   3   | 0.147 | 1.073 | 5.496 | 0.227 | 0.804 | 0.935 | 0.974 |
|   4   | 0.215 | 1.767 | 7.537 | 0.317 | 0.647 | 0.865 | 0.940 |
|   5   | 0.325 | 3.295 | 10.676 | 0.471 | 0.411 | 0.728 | 0.863 |
|  avg  | 0.187 | 1.593 | 6.749 | 0.284 | 0.712 | 0.886 | 0.947 |

- **Summary:** $\text{DEE}_1=$ 0.475, $\text{DEE}_2=$ 0.238, $\text{DEE}_3=$ 0.263


### Defocus Blur
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.156 | 1.196 | 6.278 | 0.249 | 0.774 | 0.921 | 0.966 |
|   2   | 0.219 | 1.968 | 8.232 | 0.341 | 0.634 | 0.851 | 0.928 |
|   3   | 0.295 | 3.010 | 10.060 | 0.443 | 0.489 | 0.759 | 0.872 |
|   4   | 0.326 | 3.438 | 10.640 | 0.481 | 0.440 | 0.720 | 0.848 |
|   5   | 0.321 | 3.425 | 10.718 | 0.482 | 0.449 | 0.725 | 0.846 |
|  avg  | 0.263 | 2.607 | 9.186 | 0.399 | 0.557 | 0.795 | 0.892 |

- **Summary:** $\text{DEE}_1=$ 0.706, $\text{DEE}_2=$ 0.353, $\text{DEE}_3=$ 0.473


### Glass Blur
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.137 | 1.042 | 5.652 | 0.218 | 0.825 | 0.940 | 0.975 |
|   2   | 0.163 | 1.261 | 6.472 | 0.254 | 0.764 | 0.916 | 0.963 |
|   3   | 0.219 | 1.853 | 7.984 | 0.324 | 0.628 | 0.867 | 0.936 |
|   4   | 0.241 | 2.134 | 8.628 | 0.357 | 0.570 | 0.842 | 0.922 |
|   5   | 0.302 | 2.997 | 10.145 | 0.444 | 0.455 | 0.758 | 0.876 |
|  avg  | 0.212 | 1.857 | 7.776 | 0.319 | 0.648 | 0.865 | 0.934 |

- **Summary:** $\text{DEE}_1=$ 0.564, $\text{DEE}_2=$ 0.282, $\text{DEE}_3=$ 0.328


### Motion Blur
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.127 | 0.936 | 5.241 | 0.206 | 0.840 | 0.949 | 0.979 |
|   2   | 0.150 | 1.167 | 6.137 | 0.237 | 0.784 | 0.927 | 0.971 |
|   3   | 0.191 | 1.636 | 7.492 | 0.296 | 0.691 | 0.881 | 0.949 |
|   4   | 0.231 | 2.083 | 8.472 | 0.350 | 0.601 | 0.834 | 0.923 |
|   5   | 0.248 | 2.308 | 8.899 | 0.374 | 0.570 | 0.811 | 0.911 |
|  avg  | 0.189 | 1.626 | 7.248 | 0.293 | 0.697 | 0.880 | 0.947 |

- **Summary:** $\text{DEE}_1=$ 0.492, $\text{DEE}_2=$ 0.246, $\text{DEE}_3=$ 0.272


### Zoom Blur
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.157 | 1.206 | 5.900 | 0.241 | 0.781 | 0.924 | 0.969 |
|   2   | 0.168 | 1.295 | 6.196 | 0.255 | 0.755 | 0.913 | 0.965 |
|   3   | 0.165 | 1.261 | 6.145 | 0.249 | 0.765 | 0.915 | 0.966 |
|   4   | 0.173 | 1.332 | 6.325 | 0.258 | 0.747 | 0.908 | 0.964 |
|   5   | 0.172 | 1.336 | 6.299 | 0.255 | 0.751 | 0.910 | 0.964 |
|  avg  | 0.167 | 1.286 | 6.173 | 0.252 | 0.760 | 0.914 | 0.966 |

- **Summary:** $\text{DEE}_1=$ 0.407, $\text{DEE}_2=$ 0.204, $\text{DEE}_3=$ 0.220


### Elastic Transform
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.113 | 0.833 | 4.793 | 0.190 | 0.873 | 0.958 | 0.981 |
|   2   | 0.114 | 0.834 | 4.856 | 0.192 | 0.869 | 0.957 | 0.981 |
|   3   | 0.116 | 0.849 | 4.963 | 0.195 | 0.863 | 0.955 | 0.981 |
|   4   | 0.119 | 0.868 | 5.065 | 0.198 | 0.857 | 0.952 | 0.980 |
|   5   | 0.123 | 0.908 | 5.228 | 0.204 | 0.847 | 0.949 | 0.979 |
|  avg  | 0.117 | 0.858 | 4.981 | 0.196 | 0.862 | 0.954 | 0.980 |

- **Summary:** $\text{DEE}_1=$ 0.255, $\text{DEE}_2=$ 0.128, $\text{DEE}_3=$ 0.136


### Color Quant
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.110 | 0.809 | 4.677 | 0.186 | 0.879 | 0.961 | 0.982 |
|   2   | 0.111 | 0.815 | 4.710 | 0.187 | 0.876 | 0.961 | 0.982 |
|   3   | 0.121 | 0.879 | 4.949 | 0.199 | 0.856 | 0.955 | 0.981 |
|   4   | 0.148 | 1.102 | 5.733 | 0.233 | 0.799 | 0.932 | 0.973 |
|   5   | 0.204 | 1.702 | 7.319 | 0.305 | 0.673 | 0.879 | 0.947 |
|  avg  | 0.139 | 1.061 | 5.478 | 0.222 | 0.817 | 0.938 | 0.973 |

- **Summary:** $\text{DEE}_1=$ 0.322, $\text{DEE}_2=$ 0.161, $\text{DEE}_3=$ 0.170


### Gaussian Noise
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.174 | 1.272 | 6.221 | 0.256 | 0.735 | 0.908 | 0.967 |
|   2   | 0.213 | 1.702 | 7.461 | 0.310 | 0.641 | 0.869 | 0.943 |
|   3   | 0.262 | 2.291 | 8.838 | 0.374 | 0.523 | 0.815 | 0.914 |
|   4   | 0.293 | 2.707 | 9.633 | 0.415 | 0.455 | 0.771 | 0.893 |
|   5   | 0.322 | 3.073 | 10.182 | 0.450 | 0.408 | 0.726 | 0.876 |
|  avg  | 0.253 | 2.209 | 8.467 | 0.361 | 0.552 | 0.818 | 0.919 |

- **Summary:** $\text{DEE}_1=$ 0.700, $\text{DEE}_2=$ 0.350, $\text{DEE}_3=$ 0.458


### Impulse Noise
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.171 | 1.262 | 6.166 | 0.262 | 0.741 | 0.909 | 0.963 |
|   2   | 0.216 | 1.747 | 7.623 | 0.320 | 0.629 | 0.863 | 0.939 |
|   3   | 0.247 | 2.138 | 8.576 | 0.361 | 0.555 | 0.829 | 0.920 |
|   4   | 0.281 | 2.590 | 9.515 | 0.405 | 0.481 | 0.788 | 0.898 |
|   5   | 0.300 | 2.838 | 9.935 | 0.429 | 0.446 | 0.765 | 0.888 |
|  avg  | 0.243 | 2.115 | 8.363 | 0.355 | 0.570 | 0.831 | 0.922 |

- **Summary:** $\text{DEE}_1=$ 0.673, $\text{DEE}_2=$ 0.336, $\text{DEE}_3=$ 0.426


### Shot Noise
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.144 | 1.004 | 5.543 | 0.219 | 0.799 | 0.939 | 0.979 |
|   2   | 0.183 | 1.437 | 7.040 | 0.274 | 0.707 | 0.894 | 0.960 |
|   3   | 0.238 | 2.118 | 8.718 | 0.350 | 0.575 | 0.834 | 0.924 |
|   4   | 0.287 | 2.709 | 9.758 | 0.411 | 0.466 | 0.774 | 0.893 |
|   5   | 0.312 | 2.996 | 10.109 | 0.441 | 0.427 | 0.733 | 0.877 |
|  avg  | 0.233 | 2.053 | 8.234 | 0.339 | 0.595 | 0.835 | 0.927 |

- **Summary:** $\text{DEE}_1=$ 0.638, $\text{DEE}_2=$ 0.319, $\text{DEE}_3=$ 0.391


### ISO Noise
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.205 | 1.696 | 7.700 | 0.305 | 0.658 | 0.870 | 0.945 |
|   2   | 0.226 | 1.929 | 8.220 | 0.331 | 0.604 | 0.851 | 0.933 |
|   3   | 0.255 | 2.271 | 8.910 | 0.369 | 0.536 | 0.819 | 0.915 |
|   4   | 0.281 | 2.594 | 9.497 | 0.402 | 0.477 | 0.786 | 0.899 |
|   5   | 0.305 | 2.871 | 9.923 | 0.430 | 0.434 | 0.751 | 0.885 |
|  avg  | 0.254 | 2.272 | 8.850 | 0.367 | 0.542 | 0.815 | 0.915 |

- **Summary:** $\text{DEE}_1=$ 0.713, $\text{DEE}_2=$ 0.356, $\text{DEE}_3=$ 0.470


### Pixelate
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.118 | 0.906 | 4.852 | 0.193 | 0.866 | 0.956 | 0.981 |
|   2   | 0.124 | 0.975 | 4.958 | 0.198 | 0.858 | 0.954 | 0.980 |
|   3   | 0.137 | 1.082 | 5.195 | 0.211 | 0.839 | 0.945 | 0.977 |
|   4   | 0.154 | 1.232 | 5.565 | 0.228 | 0.811 | 0.934 | 0.972 |
|   5   | 0.165 | 1.344 | 5.830 | 0.240 | 0.789 | 0.926 | 0.969 |
|  avg  | 0.140 | 1.108 | 5.280 | 0.214 | 0.833 | 0.943 | 0.976 |

- **Summary:** $\text{DEE}_1=$ 0.307, $\text{DEE}_2=$ 0.154, $\text{DEE}_3=$ 0.168


### JPEG Compression
| Level | $\text{Abs Rel}$ | $\text{Sq Rel}$ | $\text{RMSE}$ | $\text{RMSE log}$ | $\delta < 1.25$ | $\delta < 1.25^2$ | $\delta < 1.25^3$ |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
|   1   | 0.123 | 0.944 | 5.109 | 0.202 | 0.851 | 0.952 | 0.979 |
|   2   | 0.128 | 0.995 | 5.221 | 0.208 | 0.840 | 0.949 | 0.978 |
|   3   | 0.133 | 1.036 | 5.312 | 0.213 | 0.832 | 0.945 | 0.977 |
|   4   | 0.150 | 1.177 | 5.608 | 0.231 | 0.799 | 0.932 | 0.973 |
|   5   | 0.178 | 1.458 | 6.086 | 0.260 | 0.749 | 0.912 | 0.963 |
|  avg  | 0.142 | 1.122 | 5.467 | 0.223 | 0.814 | 0.938 | 0.974 |

- **Summary:** $\text{DEE}_1=$ 0.328, $\text{DEE}_2=$ 0.164, $\text{DEE}_3=$ 0.175

