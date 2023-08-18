<img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/logo2.png" align="right" width="34%">

# Experimental Result

## Outline
- [Robustness Benchmark](#robustness-benchmark)
  - [KITTI-C](#red_car-kitti-c)
  - [NYUDepth2-C](#taxi-nyudepth2-c)
  - [nuScenes-C](#blue_car-nuscenes-c)
- [Analysis & Discussion]()
  - [Idiosyncrasy Analysis](#idiosyncrasy-analysis)
  - [Severity Level](#severity-level)
  - [Model Complexity]()


## Robustness Benchmark

### :red_car: KITTI-C
| Model | Docs | Checkpoint | Modality | mCE (%) | mRR (%) | Clean | Bright | Dark | Fog | Frost | Snow | Contrast | Defocus | Glass | Motion | Zoom | Elastic| Quant| Gaussian | Impulse | Shot | ISO | Pixelate | JPEG | 
| :-- | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| MonoDepth2<sub>R18</sub><sup>:star:</sup> | [Link]() | [Download]() | Mono | 100.00 | 84.46 |  0.119 | 0.130     | 0.280     | 0.155     | 0.277     | 0.511     | 0.187     | 0.244     | 0.242     | 0.216     | 0.201     | 0.129     | 0.193     | 0.384     | 0.389     | 0.340     | 0.388     | 0.145     | 0.196     |
| MonoDepth2<sub>R18+nopt</sub> | [Link]() | [Download]() | Mono | 119.75 | 82.50 | 0.144 | 0.183     | 0.343     | 0.311     | 0.312     | 0.399     | 0.416     | 0.254     | 0.232     | 0.199     | 0.207     | 0.148     | 0.212     | 0.441     | 0.452     | 0.402     | 0.453     | 0.153     | 0.171     |
| MonoDepth2<sub>R18+HR</sub> | [Link]() | [Download]() | Mono | 106.06 | 82.44 | 0.114 | 0.129     | 0.376     | 0.155     | 0.271     | 0.582     | 0.214     | 0.393     | 0.257     | 0.230     | 0.232     | 0.123     | 0.215     | 0.326     | 0.352     | 0.317     | 0.344     | 0.138     | 0.198     |
| MonoDepth2<sub>R50</sub> | [Link]() | [Download]() | Mono | 113.43 | 80.59 | 0.117 | 0.127 | 0.294 | 0.155 | 0.287 | 0.492 | 0.233 | 0.427 | 0.392 | 0.277 | 0.208 | 0.130 | 0.198 | 0.409 | 0.403 | 0.368 | 0.425 | 0.155 | 0.211 |
| MaskOcc | [Link]() | [Download]() | Mono | 104.05 | 82.97 | 0.117 | 0.130 | 0.285 | 0.154 | 0.283 | 0.492 | 0.200 | 0.318 | 0.295 | 0.228 | 0.201 | 0.129 | 0.184 | 0.403 | 0.410 | 0.364 | 0.417 | 0.143 | 0.177 |
| DNet<sub>R18</sub> | [Link]() | [Download]() | Mono | 104.71 | 83.34 | 0.118 | 0.128 | 0.264 | 0.156 | 0.317 | 0.504 | 0.209 | 0.348 | 0.320 | 0.242 | 0.215 | 0.131 | 0.189 | 0.362 | 0.366 | 0.326 | 0.357 | 0.145 | 0.190 |
| CADepth | [Link]() | [Download]() | Mono | 110.11 | 80.07 | 0.108 | 0.121 | 0.300 | 0.142 | 0.324 | 0.529 | 0.193 | 0.356 | 0.347 | 0.285 | 0.208 | 0.121 | 0.192 | 0.423 | 0.433 | 0.383 | 0.448 | 0.144 | 0.195 |
| HR-Depth | [Link]() | [Download]() | Mono | 103.73 | 82.93 | 0.112 | 0.121 | 0.289 | 0.151 | 0.279 | 0.481 | 0.213 | 0.356 | 0.300 | 0.263 | 0.224 | 0.124 | 0.187 | 0.363 | 0.373 | 0.336 | 0.374 | 0.135 | 0.176 |
| DIFFNet<sub>HRNet</sub> | [Link]() | [Download]() | Mono | 94.96 | 85.41 | 0.102 | 0.111 | 0.222 | 0.131 | 0.199 | 0.352 | 0.161 | 0.513 | 0.330 | 0.280 | 0.197 | 0.114 | 0.165 | 0.292 | 0.266 | 0.255 | 0.270 | 0.135 | 0.202 |
| ManyDepth<sub>single</sub> | [Link]() | [Download]() | Mono | 105.41 | 83.11 | 0.123 | 0.135 | 0.274 | 0.169 | 0.288 | 0.479 | 0.227 | 0.254 | 0.279 | 0.211 | 0.194     | 0.134     | 0.189 | 0.430 | 0.450 | 0.387 | 0.452 | 0.147 | 0.182 |
| FSRE-Depth | [Link]() | [Download]() | Mono | 99.05 | 83.86 | 0.109 | 0.128 | 0.261 | 0.139 | 0.237 | 0.393 | 0.170 | 0.291 | 0.273 | 0.214 | 0.185 | 0.119 | 0.179 | 0.400 | 0.414 | 0.370 | 0.407 | 0.147 | 0.224 |
| MonoViT<sub>MPViT</sub> | [Link]() | [Download]() | Mono | 79.33 | 89.15 | 0.099 | 0.106 | 0.243 | 0.116 | 0.213 | 0.275 | 0.119 | 0.180 | 0.204 | 0.163 | 0.179 | 0.118 | 0.146 | 0.310 | 0.293 | 0.271 | 0.290 | 0.162 | 0.154 |
| MonoViT<sub>MPViT+HR</sub> | [Link]() | [Download]() | Mono | 74.95 | 89.72 | 0.094 | 0.102 | 0.238 | 0.114 | 0.225 | 0.269 | 0.117 | 0.145 | 0.171 | 0.145 | 0.184 | 0.108 | 0.145 | 0.302 | 0.277 | 0.259 | 0.285 | 0.135 | 0.148 |
| DynaDepth<sub>R18</sub> | [Link]() | [Download]() | Mono | 110.38 | 81.50 | 0.117 | 0.128 | 0.289 | 0.156 | 0.289 | 0.509 | 0.208 | 0.501 | 0.347 | 0.305 | 0.207 | 0.127 | 0.186 | 0.379 | 0.379 | 0.336 | 0.379 | 0.141 | 0.180 |
| DynaDepth<sub>R50</sub> | [Link]() | [Download]() | Mono | 119.99 | 77.98 | 0.113 | 0.128 | 0.298 | 0.152 | 0.324 | 0.549 | 0.201 | 0.532 | 0.454 | 0.318 | 0.218 | 0.125 | 0.197 | 0.418 | 0.437 | 0.382 | 0.448 | 0.153 | 0.216 |
| RA-Depth<sub>HRNet</sub> | [Link]() | [Download]() | Mono | 112.73 | 78.79 | 0.096 | 0.113 | 0.314 | 0.127 | 0.239 | 0.413 | 0.165 | 0.499 | 0.368 | 0.378 | 0.214 | 0.122 | 0.178 | 0.423 | 0.403 | 0.402 | 0.455 | 0.175 | 0.192 |
| TriDepth<sub>single</sub> | [Link]() | [Download]() | Mono | 109.26 | 81.56 | 0.117 | 0.131 | 0.300 | 0.188 | 0.338 | 0.498 | 0.265 | 0.268 | 0.301 | 0.212 | 0.190 | 0.126 | 0.199 | 0.418 | 0.438 | 0.380 | 0.438 | 0.142 | 0.205 |
| Lite-Mono<sub>Tiny</sub> | [Link]() | [Download]() | Mono | 92.92 | 86.69 | 0.115 | 0.127 | 0.257 | 0.157 | 0.225 | 0.354 | 0.191 | 0.257 | 0.248 | 0.198 | 0.186 | 0.127 | 0.159 | 0.358 | 0.342 | 0.336 | 0.360 | 0.147 | 0.161 |
| Lite-Mono<sub>Tiny+HR</sub> | [Link]() | [Download]() | Mono | 86.71 | 87.63 | 0.106 | 0.119 | 0.227 | 0.139 | 0.282 | 0.370 | 0.166 | 0.216 | 0.201 | 0.190 | 0.202 | 0.116 | 0.146 | 0.320 | 0.291 | 0.286 | 0.312 | 0.148 | 0.167 | 
| Lite-Mono<sub>Small</sub> | [Link]() | [Download]() | Mono | 100.34 | 84.67 | 0.115 | 0.127 | 0.251 | 0.162 | 0.251 | 0.430 | 0.238 | 0.353 | 0.282 | 0.246 | 0.204 | 0.128 | 0.161 | 0.350 | 0.336 | 0.319 | 0.356 | 0.154 | 0.164 |
| Lite-Mono<sub>Small+HR</sub> | [Link]() | [Download]() | Mono | 89.90 | 86.05 | 0.105 | 0.119 | 0.263 | 0.139 | 0.263 | 0.436 | 0.167 | 0.188 | 0.181 | 0.193 | 0.214 | 0.117 | 0.147 | 0.366 | 0.354 | 0.327 | 0.355 | 0.152 | 0.157 |
| Lite-Mono<sub>Base</sub> | [Link]() | [Download]() | Mono | 93.16 | 85.99 | 0.110 | 0.119 | 0.259 | 0.144 | 0.245 | 0.384 | 0.177 | 0.224 | 0.237 | 0.221 | 0.196 | 0.129 | 0.175 | 0.361 | 0.340 | 0.334 | 0.363 | 0.151 | 0.165 |
| Lite-Mono<sub>Base+HR</sub> | [Link]() | [Download]() | Mono | 89.85 | 85.80 | 0.103 | 0.115 | 0.256 | 0.135 | 0.258 | 0.486 | 0.164 | 0.220 | 0.194 | 0.213 | 0.205 | 0.114 | 0.154 | 0.340 | 0.327 | 0.321 | 0.344 | 0.145 | 0.156 |
| Lite-Mono<sub>Large</sub> | [Link]() | [Download]() | Mono | 90.75 | 85.54 | 0.102 | 0.110 | 0.227 | 0.126 | 0.255 | 0.433 | 0.149 | 0.222 | 0.225 | 0.220 | 0.192 | 0.121 | 0.148 | 0.363 | 0.348 | 0.329 | 0.362 | 0.160 | 0.184 |
| Lite-Mono<sub>Large+HR</sub> | [Link]() | [Download]() | Mono | 92.01 | 83.90 | 0.096 | 0.112 | 0.241 | 0.122 | 0.280 | 0.482 | 0.141 | 0.193 | 0.194 | 0.213 | 0.222 | 0.108 | 0.140 | 0.403 | 0.404 | 0.365 | 0.407 | 0.139 | 0.182 | 
| |
| MonoDepth2<sub>R18</sub> | [Link]() | [Download]() | Stereo | 117.69 | 79.05 | 0.123 | 0.133     | 0.348     | 0.161     | 0.305     | 0.515     | 0.234     | 0.390     | 0.332     | 0.264     | 0.209     | 0.135     | 0.200     | 0.492     | 0.509     | 0.463     | 0.493     | 0.144     | 0.194     |
| MonoDepth2<sub>R18+nopt</sub> | [Link]() | [Download]() | Stereo | 128.98 | 79.20 | 0.150 | 0.181     | 0.422     | 0.292     | 0.352     | 0.435     | 0.342     | 0.266     | 0.232     | 0.217     | 0.229     | 0.156     | 0.236     | 0.539     | 0.564     | 0.521     | 0.556     | 0.164     | 0.178     |
| MonoDepth2<sub>R18+HR</sub> | [Link]() | [Download]() | Stereo | 111.46 | 81.65 | 0.117 | 0.132     | 0.285     | 0.167     | 0.356     | 0.529     | 0.238     | 0.432     | 0.312     | 0.279     | 0.246     | 0.130     | 0.206     | 0.343     | 0.343     | 0.322     | 0.344     | 0.150     | 0.209     |
| DepthHints | [Link]() | [Download]() | Stereo | 111.41 | 80.08 |  0.113 | 0.124     | 0.310     | 0.137     | 0.321     | 0.515     | 0.164     | 0.350     | 0.410     | 0.263     | 0.196     | 0.130     | 0.192     | 0.440     | 0.447     | 0.412     | 0.455     | 0.157     | 0.192     |
| DepthHints<sub>HR</sub> | [Link]() | [Download]() | Stereo | 112.02 | 79.53 | 0.104 | 0.122     | 0.282     | 0.141     | 0.317     | 0.480     | 0.180     | 0.459     | 0.363     | 0.320     | 0.262     | 0.118     | 0.183     | 0.397     | 0.421     | 0.380     | 0.424     | 0.141     | 0.183     |
| DepthHints<sub>HR+nopt</sub> | [Link]() | [Download]() | Stereo | 141.61 | 73.18 | 0.134 | 0.173     | 0.476     | 0.301     | 0.374     | 0.463     | 0.393     | 0.357     | 0.289     | 0.241     | 0.231     | 0.142     | 0.247     | 0.613     | 0.658     | 0.599     | 0.692     | 0.152     | 0.191     |
| |
| MonoDepth2<sub>R18</sub> | [Link]() | [Download]() | M+S | 124.31 | 75.36 |  0.116 | 0.127     | 0.404     | 0.150     | 0.295     | 0.536     | 0.199     | 0.447     | 0.346     | 0.283     | 0.204     | 0.128     | 0.203     | 0.577     | 0.605     | 0.561     | 0.629     | 0.136     | 0.179     |
| MonoDepth2<sub>R18+nopt</sub> | [Link]() | [Download]() | M+S | 136.25 | 76.72 | 0.146 | 0.193     | 0.460     | 0.328     | 0.421     | 0.428     | 0.440     | 0.228     | 0.221     | 0.216     | 0.230     | 0.153     | 0.229     | 0.570     | 0.596     | 0.549     | 0.606     | 0.161     | 0.177     |
| MonoDepth2<sub>R18+HR</sub> | [Link]() | [Download]() | M+S | 106.06 | 82.44 | 0.114 | 0.129 | 0.376 | 0.155 | 0.271 | 0.582 | 0.214 | 0.393 | 0.257 | 0.230 | 0.232 | 0.123 | 0.215 | 0.326 | 0.352 | 0.317 | 0.344 | 0.138 | 0.198 |
| CADepth | [Link]() | [Download]() | M+S | 118.29 | 76.68 | 0.110 | 0.123 | 0.357 | 0.137 | 0.311 | 0.556 | 0.169 | 0.338 | 0.412 | 0.260 | 0.193 | 0.126 | 0.186 | 0.546 | 0.559 | 0.524 | 0.582 | 0.145 | 0.192 |
| MonoViT<sub>MPViT</sub> | [Link]() | [Download]() | M+S | 75.39 | 90.39 | 0.098 | 0.104 | 0.245 | 0.122 | 0.213 | 0.215 | 0.131 | 0.179 | 0.184 | 0.161 | 0.168 | 0.112 | 0.147 | 0.277 | 0.257 | 0.242 | 0.260 | 0.147 | 0.144 | 
| MonoViT<sub>MPViT+HR</sub> | [Link]() | [Download]() | M+S | 70.79 | 90.67 | 0.090 | 0.097 | 0.221 | 0.113 | 0.217 | 0.253 | 0.113 | 0.146 | 0.159 | 0.144 | 0.175 | 0.098 | 0.138 | 0.267 | 0.246 | 0.236 | 0.246 | 0.135 | 0.145 |


### :taxi: NYUDepth2-C
| Model | Docs | Checkpoint | mCE (%) | mRR (%) | Clean | Bright | Dark | Contrast | Defocus | Glass | Motion | Zoom | Elastic | Quant | Gaussian | Impulse | Shot | ISO | Pixelate | JPEG | 
| :-- | :--: |  :--: |  :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| BTS<sub>R50</sub> | [Link]() | [Download]() | 122.78 | 80.63 | 0.122 | 0.149 | 0.269 | 0.265 | 0.337 | 0.262 | 0.231 | 0.372 | 0.182 | 0.180 | 0.442 | 0.512 | 0.392 | 0.474 | 0.139 | 0.175 |
| AdaBins<sub>R50</sub> | [Link]() | [Download]() | 134.69 | 81.62 | 0.158 | 0.179 | 0.293 | 0.289 | 0.339 | 0.280 | 0.245 | 0.390 | 0.204 | 0.216 | 0.458 | 0.519 | 0.401 | 0.481 | 0.186 | 0.211 |
| AdaBins<sub>EfficientB5</sub><sup>:star:</sup> | [Link]() | [Download]() | 100.00 | 85.83 | 0.112 | 0.132 | 0.194 | 0.212 | 0.235 | 0.206 | 0.184 | 0.384 | 0.153 | 0.151 | 0.390 | 0.374 | 0.294 | 0.380 | 0.124 | 0.154 |
| DPT<sub>ViT-B</sub> | [Link]() | [Download]() | 83.22 | 95.25 | 0.136 | 0.135 | 0.182 | 0.180 | 0.154 | 0.166 | 0.155 | 0.232 | 0.139 | 0.165 | 0.200 | 0.213 | 0.191 | 0.199 | 0.171 | 0.174 |
| SimIPU<sub>R50+no_pt</sub> | [Link]() | [Download]() | 200.17 | 92.52 | 0.372 | 0.388 | 0.427 | 0.448 | 0.416 | 0.401 | 0.400 | 0.433 | 0.381 | 0.391 | 0.465 | 0.471 | 0.450 | 0.461 | 0.375 | 0.378 |
| SimIPU<sub>R50+imagenet</sub> | [Link]() | [Download]() | 163.06 | 85.01 | 0.244 | 0.269 | 0.370 | 0.376 | 0.377 | 0.337 | 0.324 | 0.422 | 0.306 | 0.289 | 0.445 | 0.463 | 0.414 | 0.449 | 0.247 | 0.272 |
| SimIPU<sub>R50+kitti</sub> | [Link]() | [Download]() | 173.78 | 91.64 | 0.312 | 0.326 | 0.373 | 0.406 | 0.360 | 0.333 | 0.335 | 0.386 | 0.316 | 0.333 | 0.432 | 0.442 | 0.422 | 0.443 | 0.314 | 0.322 |
| SimIPU<sub>R50+waymo</sub> | [Link]() | [Download]() | 159.46 | 85.73 | 0.243 | 0.269 | 0.348 | 0.398 | 0.380 | 0.327 | 0.313 | 0.405 | 0.256 | 0.287 | 0.439 | 0.461 | 0.416 | 0.455 | 0.246 | 0.265 |
| DepthFormer<sub>SwinT_w7_1k</sub> | [Link]() | [Download]() | 106.34 | 87.25 | 0.125 | 0.147 | 0.279 | 0.235 | 0.220 | 0.260 | 0.191 | 0.300 | 0.175 | 0.192 | 0.294 | 0.321 | 0.289 | 0.305 | 0.161 | 0.179 |
| DepthFormer<sub>SwinT_w7_22k</sub> | [Link]() | [Download]() | 63.47 | 94.19 | 0.086 | 0.099 | 0.150 | 0.123 | 0.127 | 0.172 | 0.119 | 0.237 | 0.112 | 0.119 | 0.159 | 0.156 | 0.148 | 0.157 | 0.101 | 0.108 |


### :blue_car: nuScenes-C
| Model | Docs | Checkpoint | Sclae | Resolution | Clean | Bright | Dark | Fog | Snow | Motion | Quant |
| :-- | :--: |  :--: |  :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |
| MonoDepth2<sup>:star:</sup> | [Link](https://github.com/nianticlabs/monodepth2) | [Download](https://storage.googleapis.com/niantic-lon-static/research/monodepth2/mono_640x192.zip) | Ambiguous | 640 x 192 | 0.304 | 0.327 | 0.355 | 0.353 | 0.436 | 0.335 | 0.306 |
| SurroundDepth | [Link](https://github.com/weiyithu/SurroundDepth) | [Download](https://cloud.tsinghua.edu.cn/f/72717fa447f749e38480/?dl=1) | Ambiguous | 640 x 384 | 0.245 | 0.291 | 0.294 | 0.265 | 0.325 | 0.265 | 0.287 |
| SurroundDepth | [Link](https://github.com/weiyithu/SurroundDepth) | [Download](https://cloud.tsinghua.edu.cn/f/caad458a790c48e380d4/?dl=1) | Aware     | 640 x 384 | 0.271 | 0.313 | 0.366 | 0.323 | 0.449 | 0.364 | 0.320 |


## Analysis & Discussion

### Idiosyncrasy Analysis
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/idiosyncrasy/category.jpg" width="560"> |
| :-: |

| | | | |
| :-: | :-: | :-: | :-: | 
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/idiosyncrasy/MonoDepth2.jpg" width="210"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/idiosyncrasy/CADepth.jpg" width="210"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/idiosyncrasy/DIFFNet.jpg" width="210"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/idiosyncrasy/DNet.jpg" width="210"> |
| |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/idiosyncrasy/DepthHints.jpg" width="210"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/idiosyncrasy/FSRE-Depth.jpg" width="210"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/idiosyncrasy/HR-Depth.jpg" width="210"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/idiosyncrasy/ManyDepth.jpg" width="210"> |
| |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/idiosyncrasy/MaskOcc.jpg" width="210"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/idiosyncrasy/MonoViT.jpg" width="210"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/idiosyncrasy/Lite-Mono.jpg" width="210"> | <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/idiosyncrasy/RA-Depth.jpg" width="210"> |
| |


### Severity Level
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/models.jpg" width="800"> |
| :-: |

| | | |
| :-: | :-: | :-: | 
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/brightness.jpg" width="300"> |  <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/dark.jpg" width="300"> |  <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/fog.jpg" width="300"> |
| |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/frost.jpg" width="300"> |  <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/snow.jpg" width="300"> |  <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/contrast.jpg" width="300"> |
| |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/defocus_blur.jpg" width="300"> |  <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/glass_blur.jpg" width="300"> |  <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/motion_blur.jpg" width="300"> |
| |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/zoom_blur.jpg" width="300"> |  <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/elastic_transform.jpg" width="300"> |  <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/color_quant.jpg" width="300"> |
| |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/gaussian_noise.jpg" width="300"> |  <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/impulse_noise.jpg" width="300"> |  <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/shot_noise.jpg" width="300"> |
| |
| <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/iso_noise.jpg" width="300"> |  <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/pixelate.jpg" width="300"> |  <img src="https://github.com/ldkong1205/RoboDepth/blob/main/docs/figs/severity_level/jpeg_compression.jpg" width="300"> |
| |

