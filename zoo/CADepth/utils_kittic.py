import numpy as np
import PIL
from PIL import Image
from torchvision import transforms
from imagecorruptions import corrupt


def get_scores_monodepth2_mono_old():  # tested with the on-the-fly mode, now we are using stored images
    baseline = {
        'DES1': dict(), 'DES2': dict(), 'DES3': dict()
    }

    # DES1 = abs_rel - a1 + 1
    baseline['DES1']['clean']               = 0.238
    baseline['DES1']['brightness']          = 0.259
    baseline['DES1']['dark']                = 0.562
    baseline['DES1']['fog']                 = 0.310
    baseline['DES1']['frost']               = 0.553
    baseline['DES1']['snow']                = 1.024
    baseline['DES1']['contrast']            = 0.373
    baseline['DES1']['defocus_blur']        = 0.487
    baseline['DES1']['glass_blur']          = 0.484
    baseline['DES1']['motion_blur']         = 0.437
    baseline['DES1']['zoom_blur']           = 0.402
    baseline['DES1']['elastic_transform']   = 0.258
    baseline['DES1']['color_quant']         = 0.385
    baseline['DES1']['gaussian_noise']      = 0.768
    baseline['DES1']['impulse_noise']       = 0.778
    baseline['DES1']['shot_noise']          = 0.681
    baseline['DES1']['iso_noise']           = 0.775
    baseline['DES1']['pixelate']            = 0.289
    baseline['DES1']['jpeg_compression']    = 0.391

    # DES2 = 0.5 * (abs_rel - a1 + 1)
    baseline['DES2']['clean']               = 0.119
    baseline['DES2']['brightness']          = 0.130
    baseline['DES2']['dark']                = 0.281
    baseline['DES2']['fog']                 = 0.155
    baseline['DES2']['frost']               = 0.277
    baseline['DES2']['snow']                = 0.512
    baseline['DES2']['contrast']            = 0.186
    baseline['DES2']['defocus_blur']        = 0.244	
    baseline['DES2']['glass_blur']          = 0.242
    baseline['DES2']['motion_blur']         = 0.218
    baseline['DES2']['zoom_blur']           = 0.201	
    baseline['DES2']['elastic_transform']   = 0.129
    baseline['DES2']['color_quant']         = 0.193
    baseline['DES2']['gaussian_noise']      = 0.384
    baseline['DES2']['impulse_noise']       = 0.389
    baseline['DES2']['shot_noise']          = 0.340
    baseline['DES2']['iso_noise']           = 0.388
    baseline['DES2']['pixelate']            = 0.145
    baseline['DES2']['jpeg_compression']    = 0.196

    # DES3 = abs_rel / a1
    baseline['DES3']['clean']               = 0.131
    baseline['DES3']['brightness']          = 0.140
    baseline['DES3']['dark']                = 0.331
    baseline['DES3']['fog']                 = 0.165
    baseline['DES3']['frost']               = 0.327
    baseline['DES3']['snow']                = 1.063
    baseline['DES3']['contrast']            = 0.199
    baseline['DES3']['defocus_blur']        = 0.271
    baseline['DES3']['glass_blur']          = 0.270
    baseline['DES3']['motion_blur']         = 0.239
    baseline['DES3']['zoom_blur']           = 0.221
    baseline['DES3']['elastic_transform']   = 0.140
    baseline['DES3']['color_quant']         = 0.209
    baseline['DES3']['gaussian_noise']      = 0.551
    baseline['DES3']['impulse_noise']       = 0.562
    baseline['DES3']['shot_noise']          = 0.443
    baseline['DES3']['iso_noise']           = 0.556
    baseline['DES3']['pixelate']            = 0.158
    baseline['DES3']['jpeg_compression']    = 0.215

    return baseline


def get_scores_monodepth2_mono():
    baseline = {
        'DES1': dict(), 'DES2': dict(), 'DES3': dict()
    }

    # DES1 = abs_rel - a1 + 1
    baseline['DES1']['clean']               = 0.238
    baseline['DES1']['brightness']          = 0.259
    baseline['DES1']['dark']                = 0.561
    baseline['DES1']['fog']                 = 0.311
    baseline['DES1']['frost']               = 0.553
    baseline['DES1']['snow']                = 1.023
    baseline['DES1']['contrast']            = 0.373
    baseline['DES1']['defocus_blur']        = 0.487
    baseline['DES1']['glass_blur']          = 0.484
    baseline['DES1']['motion_blur']         = 0.433	
    baseline['DES1']['zoom_blur']           = 0.402
    baseline['DES1']['elastic_transform']   = 0.258
    baseline['DES1']['color_quant']         = 0.386
    baseline['DES1']['gaussian_noise']      = 0.768
    baseline['DES1']['impulse_noise']       = 0.779
    baseline['DES1']['shot_noise']          = 0.681	
    baseline['DES1']['iso_noise']           = 0.776
    baseline['DES1']['pixelate']            = 0.289
    baseline['DES1']['jpeg_compression']    = 0.391

    # DES2 = 0.5 * (abs_rel - a1 + 1)
    baseline['DES2']['clean']               = 0.119
    baseline['DES2']['brightness']          = 0.130
    baseline['DES2']['dark']                = 0.280
    baseline['DES2']['fog']                 = 0.155
    baseline['DES2']['frost']               = 0.277
    baseline['DES2']['snow']                = 0.511
    baseline['DES2']['contrast']            = 0.187
    baseline['DES2']['defocus_blur']        = 0.244	
    baseline['DES2']['glass_blur']          = 0.242
    baseline['DES2']['motion_blur']         = 0.216
    baseline['DES2']['zoom_blur']           = 0.201	
    baseline['DES2']['elastic_transform']   = 0.129
    baseline['DES2']['color_quant']         = 0.193
    baseline['DES2']['gaussian_noise']      = 0.384
    baseline['DES2']['impulse_noise']       = 0.389
    baseline['DES2']['shot_noise']          = 0.340
    baseline['DES2']['iso_noise']           = 0.388
    baseline['DES2']['pixelate']            = 0.145
    baseline['DES2']['jpeg_compression']    = 0.196

    # DES3 = abs_rel / a1
    baseline['DES3']['clean']               = 0.131
    baseline['DES3']['brightness']          = 0.140
    baseline['DES3']['dark']                = 0.330
    baseline['DES3']['fog']                 = 0.166
    baseline['DES3']['frost']               = 0.327
    baseline['DES3']['snow']                = 1.058
    baseline['DES3']['contrast']            = 0.199
    baseline['DES3']['defocus_blur']        = 0.271
    baseline['DES3']['glass_blur']          = 0.270
    baseline['DES3']['motion_blur']         = 0.236
    baseline['DES3']['zoom_blur']           = 0.221
    baseline['DES3']['elastic_transform']   = 0.141
    baseline['DES3']['color_quant']         = 0.209
    baseline['DES3']['gaussian_noise']      = 0.551
    baseline['DES3']['impulse_noise']       = 0.563
    baseline['DES3']['shot_noise']          = 0.442
    baseline['DES3']['iso_noise']           = 0.557
    baseline['DES3']['pixelate']            = 0.158
    baseline['DES3']['jpeg_compression']    = 0.215

    return baseline


def get_scores_monodepth2_stereo():
    baseline = {
        'DES1': dict(), 'DES2': dict(), 'DES3': dict()
    }

    # DES1 = abs_rel - a1 + 1
    baseline['DES1']['clean']               = 0.246
    baseline['DES1']['brightness']          = 0.266
    baseline['DES1']['dark']                = 0.696
    baseline['DES1']['fog']                 = 0.322
    baseline['DES1']['frost']               = 0.610
    baseline['DES1']['snow']                = 1.031
    baseline['DES1']['contrast']            = 0.468
    baseline['DES1']['defocus_blur']        = 0.779
    baseline['DES1']['glass_blur']          = 0.663
    baseline['DES1']['motion_blur']         = 0.529	
    baseline['DES1']['zoom_blur']           = 0.419
    baseline['DES1']['elastic_transform']   = 0.270
    baseline['DES1']['color_quant']         = 0.399
    baseline['DES1']['gaussian_noise']      = 0.985
    baseline['DES1']['impulse_noise']       = 1.018
    baseline['DES1']['shot_noise']          = 0.927	
    baseline['DES1']['iso_noise']           = 0.985	
    baseline['DES1']['pixelate']            = 0.288
    baseline['DES1']['jpeg_compression']    = 0.387

    # DES2 = 0.5 * (abs_rel - a1 + 1)
    baseline['DES2']['clean']               = 0.123
    baseline['DES2']['brightness']          = 0.133
    baseline['DES2']['dark']                = 0.348
    baseline['DES2']['fog']                 = 0.161
    baseline['DES2']['frost']               = 0.305
    baseline['DES2']['snow']                = 0.515
    baseline['DES2']['contrast']            = 0.234
    baseline['DES2']['defocus_blur']        = 0.390	
    baseline['DES2']['glass_blur']          = 0.332
    baseline['DES2']['motion_blur']         = 0.264
    baseline['DES2']['zoom_blur']           = 0.209
    baseline['DES2']['elastic_transform']   = 0.135
    baseline['DES2']['color_quant']         = 0.200
    baseline['DES2']['gaussian_noise']      = 0.492
    baseline['DES2']['impulse_noise']       = 0.509
    baseline['DES2']['shot_noise']          = 0.463
    baseline['DES2']['iso_noise']           = 0.493
    baseline['DES2']['pixelate']            = 0.144
    baseline['DES2']['jpeg_compression']    = 0.194

    # DES3 = abs_rel / a1
    baseline['DES3']['clean']               = 0.127
    baseline['DES3']['brightness']          = 0.137
    baseline['DES3']['dark']                = 0.455
    baseline['DES3']['fog']                 = 0.163
    baseline['DES3']['frost']               = 0.373
    baseline['DES3']['snow']                = 1.085
    baseline['DES3']['contrast']            = 0.256
    baseline['DES3']['defocus_blur']        = 0.567
    baseline['DES3']['glass_blur']          = 0.418
    baseline['DES3']['motion_blur']         = 0.298
    baseline['DES3']['zoom_blur']           = 0.222
    baseline['DES3']['elastic_transform']   = 0.138
    baseline['DES3']['color_quant']         = 0.205
    baseline['DES3']['gaussian_noise']      = 0.960
    baseline['DES3']['impulse_noise']       = 1.052
    baseline['DES3']['shot_noise']          = 0.824
    baseline['DES3']['iso_noise']           = 0.960
    baseline['DES3']['pixelate']            = 0.151
    baseline['DES3']['jpeg_compression']    = 0.202

    return baseline


def get_scores_monodepth2_mono_stereo():
    baseline = {
        'DES1': dict(), 'DES2': dict(), 'DES3': dict()
    }

    # DES1 = abs_rel - a1 + 1
    baseline['DES1']['clean']               = 0.232
    baseline['DES1']['brightness']          = 0.255
    baseline['DES1']['dark']                = 0.808
    baseline['DES1']['fog']                 = 0.300
    baseline['DES1']['frost']               = 0.589
    baseline['DES1']['snow']                = 1.073
    baseline['DES1']['contrast']            = 0.398
    baseline['DES1']['defocus_blur']        = 0.895
    baseline['DES1']['glass_blur']          = 0.692
    baseline['DES1']['motion_blur']         = 0.565	
    baseline['DES1']['zoom_blur']           = 0.408
    baseline['DES1']['elastic_transform']   = 0.256
    baseline['DES1']['color_quant']         = 0.407
    baseline['DES1']['gaussian_noise']      = 1.154
    baseline['DES1']['impulse_noise']       = 1.209
    baseline['DES1']['shot_noise']          = 1.121
    baseline['DES1']['iso_noise']           = 1.258
    baseline['DES1']['pixelate']            = 0.271
    baseline['DES1']['jpeg_compression']    = 0.358

    # DES2 = 0.5 * (abs_rel - a1 + 1)
    baseline['DES2']['clean']               = 0.116
    baseline['DES2']['brightness']          = 0.127
    baseline['DES2']['dark']                = 0.404
    baseline['DES2']['fog']                 = 0.150
    baseline['DES2']['frost']               = 0.295
    baseline['DES2']['snow']                = 0.536
    baseline['DES2']['contrast']            = 0.199
    baseline['DES2']['defocus_blur']        = 0.447
    baseline['DES2']['glass_blur']          = 0.346
    baseline['DES2']['motion_blur']         = 0.283
    baseline['DES2']['zoom_blur']           = 0.204
    baseline['DES2']['elastic_transform']   = 0.128
    baseline['DES2']['color_quant']         = 0.203
    baseline['DES2']['gaussian_noise']      = 0.577
    baseline['DES2']['impulse_noise']       = 0.605
    baseline['DES2']['shot_noise']          = 0.560
    baseline['DES2']['iso_noise']           = 0.629
    baseline['DES2']['pixelate']            = 0.136
    baseline['DES2']['jpeg_compression']    = 0.179

    # DES3 = abs_rel / a1
    baseline['DES3']['clean']               = 0.121
    baseline['DES3']['brightness']          = 0.131
    baseline['DES3']['dark']                = 0.607
    baseline['DES3']['fog']                 = 0.153
    baseline['DES3']['frost']               = 0.348
    baseline['DES3']['snow']                = 1.214
    baseline['DES3']['contrast']            = 0.205
    baseline['DES3']['defocus_blur']        = 0.755
    baseline['DES3']['glass_blur']          = 0.446
    baseline['DES3']['motion_blur']         = 0.325
    baseline['DES3']['zoom_blur']           = 0.216
    baseline['DES3']['elastic_transform']   = 0.131
    baseline['DES3']['color_quant']         = 0.210
    baseline['DES3']['gaussian_noise']      = 1.544
    baseline['DES3']['impulse_noise']       = 1.872
    baseline['DES3']['shot_noise']          = 1.398
    baseline['DES3']['iso_noise']           = 2.293
    baseline['DES3']['pixelate']            = 0.142
    baseline['DES3']['jpeg_compression']    = 0.184

    return baseline


def logger_summary(info, summary, baseline):

    # DES1 = abs_rel - a1 + 1
    clean1    = info['clean']['DES1']                      # 00
    bright1   = summary['avgs_DES1']['brightness']         # 01
    dark1     = summary['avgs_DES1']['dark']               # 02
    fog1      = summary['avgs_DES1']['fog']                # 03
    frost1    = summary['avgs_DES1']['frost']              # 04
    snow1     = summary['avgs_DES1']['snow']               # 05
    contrast1 = summary['avgs_DES1']['contrast']           # 06
    defocus1  = summary['avgs_DES1']['defocus_blur']       # 07
    glass1    = summary['avgs_DES1']['glass_blur']         # 08
    motion1   = summary['avgs_DES1']['motion_blur']        # 09
    zoom1     = summary['avgs_DES1']['zoom_blur']          # 10
    elastic1  = summary['avgs_DES1']['elastic_transform']  # 11
    quant1    = summary['avgs_DES1']['color_quant']        # 12
    gaussian1 = summary['avgs_DES1']['gaussian_noise']     # 13
    impulse1  = summary['avgs_DES1']['impulse_noise']      # 14
    shot1     = summary['avgs_DES1']['shot_noise']         # 15
    iso1      = summary['avgs_DES1']['iso_noise']          # 16
    pixelate1 = summary['avgs_DES1']['pixelate']           # 17
    jpeg1     = summary['avgs_DES1']['jpeg_compression']   # 18

    # DES2 = 0.5 * (abs_rel - a1 + 1)
    clean2    = info['clean']['DES2']                      # 00 -
    bright2   = summary['avgs_DES2']['brightness']         # 01
    dark2     = summary['avgs_DES2']['dark']               # 02
    fog2      = summary['avgs_DES2']['fog']                # 03
    frost2    = summary['avgs_DES2']['frost']              # 04
    snow2     = summary['avgs_DES2']['snow']               # 05
    contrast2 = summary['avgs_DES2']['contrast']           # 06
    defocus2  = summary['avgs_DES2']['defocus_blur']       # 07
    glass2    = summary['avgs_DES2']['glass_blur']         # 08
    motion2   = summary['avgs_DES2']['motion_blur']        # 09
    zoom2     = summary['avgs_DES2']['zoom_blur']          # 10
    elastic2  = summary['avgs_DES2']['elastic_transform']  # 11
    quant2    = summary['avgs_DES2']['color_quant']        # 12
    gaussian2 = summary['avgs_DES2']['gaussian_noise']     # 13
    impulse2  = summary['avgs_DES2']['impulse_noise']      # 14
    shot2     = summary['avgs_DES2']['shot_noise']         # 15
    iso2      = summary['avgs_DES2']['iso_noise']          # 16
    pixelate2 = summary['avgs_DES2']['pixelate']           # 17
    jpeg2     = summary['avgs_DES2']['jpeg_compression']   # 18

    # DES3 = abs_rel / a1
    clean3    = info['clean']['DES3']                      # 00
    bright3   = summary['avgs_DES3']['brightness']         # 01
    dark3     = summary['avgs_DES3']['dark']               # 02
    fog3      = summary['avgs_DES3']['fog']                # 03
    frost3    = summary['avgs_DES3']['frost']              # 04
    snow3     = summary['avgs_DES3']['snow']               # 05
    contrast3 = summary['avgs_DES3']['contrast']           # 06
    defocus3  = summary['avgs_DES3']['defocus_blur']       # 07
    glass3    = summary['avgs_DES3']['glass_blur']         # 08
    motion3   = summary['avgs_DES3']['motion_blur']        # 09
    zoom3     = summary['avgs_DES3']['zoom_blur']          # 10
    elastic3  = summary['avgs_DES3']['elastic_transform']  # 11
    quant3    = summary['avgs_DES3']['color_quant']        # 12
    gaussian3 = summary['avgs_DES3']['gaussian_noise']     # 13
    impulse3  = summary['avgs_DES3']['impulse_noise']      # 14
    shot3     = summary['avgs_DES3']['shot_noise']         # 15
    iso3      = summary['avgs_DES3']['iso_noise']          # 16
    pixelate3 = summary['avgs_DES3']['pixelate']           # 17
    jpeg3     = summary['avgs_DES3']['jpeg_compression']   # 18

    # CE (DES1)
    ce1_bright   = bright1   / baseline['DES1']['brightness']
    ce1_dark     = dark1     / baseline['DES1']['dark']
    ce1_fog      = fog1      / baseline['DES1']['fog']
    ce1_frost    = frost1    / baseline['DES1']['frost']
    ce1_snow     = snow1     / baseline['DES1']['snow']
    ce1_contrast = contrast1 / baseline['DES1']['contrast']
    ce1_defocus  = defocus1  / baseline['DES1']['defocus_blur']
    ce1_glass    = glass1    / baseline['DES1']['glass_blur']
    ce1_motion   = motion1   / baseline['DES1']['motion_blur']
    ce1_zoom     = zoom1     / baseline['DES1']['zoom_blur']
    ce1_elastic  = elastic1  / baseline['DES1']['elastic_transform']
    ce1_quant    = quant1    / baseline['DES1']['color_quant']
    ce1_gaussian = gaussian1 / baseline['DES1']['gaussian_noise']
    ce1_impulse  = impulse1  / baseline['DES1']['impulse_noise']
    ce1_shot     = shot1     / baseline['DES1']['shot_noise']
    ce1_iso      = iso1      / baseline['DES1']['iso_noise']
    ce1_pixelate = pixelate1 / baseline['DES1']['pixelate']
    ce1_jpeg     = jpeg1     / baseline['DES1']['jpeg_compression']

    # CE (DES2)
    ce2_bright   = bright2   / baseline['DES2']['brightness']
    ce2_dark     = dark2     / baseline['DES2']['dark']
    ce2_fog      = fog2      / baseline['DES2']['fog']
    ce2_frost    = frost2    / baseline['DES2']['frost']
    ce2_snow     = snow2     / baseline['DES2']['snow']
    ce2_contrast = contrast2 / baseline['DES2']['contrast']
    ce2_defocus  = defocus2  / baseline['DES2']['defocus_blur']
    ce2_glass    = glass2    / baseline['DES2']['glass_blur']
    ce2_motion   = motion2   / baseline['DES2']['motion_blur']
    ce2_zoom     = zoom2     / baseline['DES2']['zoom_blur']
    ce2_elastic  = elastic2  / baseline['DES2']['elastic_transform']
    ce2_quant    = quant2    / baseline['DES2']['color_quant']
    ce2_gaussian = gaussian2 / baseline['DES2']['gaussian_noise']
    ce2_impulse  = impulse2  / baseline['DES2']['impulse_noise']
    ce2_shot     = shot2     / baseline['DES2']['shot_noise']
    ce2_iso      = iso2      / baseline['DES2']['iso_noise']
    ce2_pixelate = pixelate2 / baseline['DES2']['pixelate']
    ce2_jpeg     = jpeg2     / baseline['DES2']['jpeg_compression']

    # CE (DES3)
    ce3_bright   = bright3   / baseline['DES3']['brightness']
    ce3_dark     = dark3     / baseline['DES3']['dark']
    ce3_fog      = fog3      / baseline['DES3']['fog']
    ce3_frost    = frost3    / baseline['DES3']['frost']
    ce3_snow     = snow3     / baseline['DES3']['snow']
    ce3_contrast = contrast3 / baseline['DES3']['contrast']
    ce3_defocus  = defocus3  / baseline['DES3']['defocus_blur']
    ce3_glass    = glass3    / baseline['DES3']['glass_blur']
    ce3_motion   = motion3   / baseline['DES3']['motion_blur']
    ce3_zoom     = zoom3     / baseline['DES3']['zoom_blur']
    ce3_elastic  = elastic3  / baseline['DES3']['elastic_transform']
    ce3_quant    = quant3    / baseline['DES3']['color_quant']
    ce3_gaussian = gaussian3 / baseline['DES3']['gaussian_noise']
    ce3_impulse  = impulse3  / baseline['DES3']['impulse_noise']
    ce3_shot     = shot3     / baseline['DES3']['shot_noise']
    ce3_iso      = iso3      / baseline['DES3']['iso_noise']
    ce3_pixelate = pixelate3 / baseline['DES3']['pixelate']
    ce3_jpeg     = jpeg3     / baseline['DES3']['jpeg_compression']

    # RCE (DES1)
    rce1_bright    = (clean1 - bright1)   / (baseline['DES1']['clean'] - baseline['DES1']['brightness'])
    rce1_dark      = (clean1 - dark1)     / (baseline['DES1']['clean'] - baseline['DES1']['dark'])
    rce1_fog       = (clean1 - fog1)      / (baseline['DES1']['clean'] - baseline['DES1']['fog'])
    rce1_frost     = (clean1 - frost1)    / (baseline['DES1']['clean'] - baseline['DES1']['frost'])
    rce1_snow      = (clean1 - snow1)     / (baseline['DES1']['clean'] - baseline['DES1']['snow'])
    rce1_contrast  = (clean1 - contrast1) / (baseline['DES1']['clean'] - baseline['DES1']['contrast'])
    rce1_defocus   = (clean1 - defocus1)  / (baseline['DES1']['clean'] - baseline['DES1']['defocus_blur'])
    rce1_glass     = (clean1 - glass1)    / (baseline['DES1']['clean'] - baseline['DES1']['glass_blur'])
    rce1_motion    = (clean1 - motion1)   / (baseline['DES1']['clean'] - baseline['DES1']['motion_blur'])
    rce1_zoom      = (clean1 - zoom1)     / (baseline['DES1']['clean'] - baseline['DES1']['zoom_blur'])
    rce1_elastic   = (clean1 - elastic1)  / (baseline['DES1']['clean'] - baseline['DES1']['elastic_transform'])
    rce1_quant     = (clean1 - quant1)    / (baseline['DES1']['clean'] - baseline['DES1']['color_quant'])
    rce1_gaussian  = (clean1 - gaussian1) / (baseline['DES1']['clean'] - baseline['DES1']['gaussian_noise'])
    rce1_impulse   = (clean1 - impulse1)  / (baseline['DES1']['clean'] - baseline['DES1']['impulse_noise'])
    rce1_shot      = (clean1 - shot1)     / (baseline['DES1']['clean'] - baseline['DES1']['shot_noise'])
    rce1_iso       = (clean1 - iso1)      / (baseline['DES1']['clean'] - baseline['DES1']['iso_noise'])
    rce1_pixelate  = (clean1 - pixelate1) / (baseline['DES1']['clean'] - baseline['DES1']['pixelate'])
    rce1_jpeg      = (clean1 - jpeg1)     / (baseline['DES1']['clean'] - baseline['DES1']['jpeg_compression'])

    # RCE (DES2)
    rce2_bright    = (clean2 - bright2)   / (baseline['DES2']['clean'] - baseline['DES2']['brightness'])
    rce2_dark      = (clean2 - dark2)     / (baseline['DES2']['clean'] - baseline['DES2']['dark'])
    rce2_fog       = (clean2 - fog2)      / (baseline['DES2']['clean'] - baseline['DES2']['fog'])
    rce2_frost     = (clean2 - frost2)    / (baseline['DES2']['clean'] - baseline['DES2']['frost'])
    rce2_snow      = (clean2 - snow2)     / (baseline['DES2']['clean'] - baseline['DES2']['snow'])
    rce2_contrast  = (clean2 - contrast2) / (baseline['DES2']['clean'] - baseline['DES2']['contrast'])
    rce2_defocus   = (clean2 - defocus2)  / (baseline['DES2']['clean'] - baseline['DES2']['defocus_blur'])
    rce2_glass     = (clean2 - glass2)    / (baseline['DES2']['clean'] - baseline['DES2']['glass_blur'])
    rce2_motion    = (clean2 - motion2)   / (baseline['DES2']['clean'] - baseline['DES2']['motion_blur'])
    rce2_zoom      = (clean2 - zoom2)     / (baseline['DES2']['clean'] - baseline['DES2']['zoom_blur'])
    rce2_elastic   = (clean2 - elastic2)  / (baseline['DES2']['clean'] - baseline['DES2']['elastic_transform'])
    rce2_quant     = (clean2 - quant2)    / (baseline['DES2']['clean'] - baseline['DES2']['color_quant'])
    rce2_gaussian  = (clean2 - gaussian2) / (baseline['DES2']['clean'] - baseline['DES2']['gaussian_noise'])
    rce2_impulse   = (clean2 - impulse2)  / (baseline['DES2']['clean'] - baseline['DES2']['impulse_noise'])
    rce2_shot      = (clean2 - shot2)     / (baseline['DES2']['clean'] - baseline['DES2']['shot_noise'])
    rce2_iso       = (clean2 - iso2)      / (baseline['DES2']['clean'] - baseline['DES2']['iso_noise'])
    rce2_pixelate  = (clean2 - pixelate2) / (baseline['DES2']['clean'] - baseline['DES2']['pixelate'])
    rce2_jpeg      = (clean2 - jpeg2)     / (baseline['DES2']['clean'] - baseline['DES2']['jpeg_compression'])

    # RCE (DES3)
    rce3_bright    = (clean3 - bright3)   / (baseline['DES3']['clean'] - baseline['DES3']['brightness'])
    rce3_dark      = (clean3 - dark3)     / (baseline['DES3']['clean'] - baseline['DES3']['dark'])
    rce3_fog       = (clean3 - fog3)      / (baseline['DES3']['clean'] - baseline['DES3']['fog'])
    rce3_frost     = (clean3 - frost3)    / (baseline['DES3']['clean'] - baseline['DES3']['frost'])
    rce3_snow      = (clean3 - snow3)     / (baseline['DES3']['clean'] - baseline['DES3']['snow'])
    rce3_contrast  = (clean3 - contrast3) / (baseline['DES3']['clean'] - baseline['DES3']['contrast'])
    rce3_defocus   = (clean3 - defocus3)  / (baseline['DES3']['clean'] - baseline['DES3']['defocus_blur'])
    rce3_glass     = (clean3 - glass3)    / (baseline['DES3']['clean'] - baseline['DES3']['glass_blur'])
    rce3_motion    = (clean3 - motion3)   / (baseline['DES3']['clean'] - baseline['DES3']['motion_blur'])
    rce3_zoom      = (clean3 - zoom3)     / (baseline['DES3']['clean'] - baseline['DES3']['zoom_blur'])
    rce3_elastic   = (clean3 - elastic3)  / (baseline['DES3']['clean'] - baseline['DES3']['elastic_transform'])
    rce3_quant     = (clean3 - quant3)    / (baseline['DES3']['clean'] - baseline['DES3']['color_quant'])
    rce3_gaussian  = (clean3 - gaussian3) / (baseline['DES3']['clean'] - baseline['DES3']['gaussian_noise'])
    rce3_impulse   = (clean3 - impulse3)  / (baseline['DES3']['clean'] - baseline['DES3']['impulse_noise'])
    rce3_shot      = (clean3 - shot3)     / (baseline['DES3']['clean'] - baseline['DES3']['shot_noise'])
    rce3_iso       = (clean3 - iso3)      / (baseline['DES3']['clean'] - baseline['DES3']['iso_noise'])
    rce3_pixelate  = (clean3 - pixelate3) / (baseline['DES3']['clean'] - baseline['DES3']['pixelate'])
    rce3_jpeg      = (clean3 - jpeg3)     / (baseline['DES3']['clean'] - baseline['DES3']['jpeg_compression'])

    # summary
    printsummary = \
"""\n
###
| Corruption | Clean | Bright | Dark | Fog | Frost | Snow | Contrast | Defocus | Glass | Motion | Zoom | Elastic| Quant| Gaussian | Impulse | Shot | ISO | Pixelate | JPEG | 
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |          
| $\\text{{DES}}_{{1}}$ | {clean1:.3f} | {bright1:.3f}     | {dark1:.3f}     | {fog1:.3f}     | {frost1:.3f}     | {snow1:.3f}     | {contrast1:.3f}     | {defocus1:.3f}     | {glass1:.3f}     | {motion1:.3f}     | {zoom1:.3f}     | {elastic1:.3f}     | {quant1:.3f}     | {gaussian1:.3f}     | {impulse1:.3f}     | {shot1:.3f}     | {iso1:.3f}     | {pixelate1:.3f}     | {jpeg1:.3f}     |
| $\\text{{CE}}_{{1}}$  |      -       | {ce1_bright:.3f}  | {ce1_dark:.3f}  | {ce1_fog:.3f}  | {ce1_frost:.3f}  | {ce1_snow:.3f}  | {ce1_contrast:.3f}  | {ce1_defocus:.3f}  | {ce1_glass:.3f}  | {ce1_motion:.3f}  | {ce1_zoom:.3f}  | {ce1_elastic:.3f}  | {ce1_quant:.3f}  | {ce1_gaussian:.3f}  | {ce1_impulse:.3f}  | {ce1_shot:.3f}  | {ce1_iso:.3f}  | {ce1_pixelate:.3f}  | {ce1_jpeg:.3f}  |
| $\\text{{RCE}}_{{1}}$ |      -       | {rce1_bright:.3f} | {rce1_dark:.3f} | {rce1_fog:.3f} | {rce1_frost:.3f} | {rce1_snow:.3f} | {rce1_contrast:.3f} | {rce1_defocus:.3f} | {rce1_glass:.3f} | {rce1_motion:.3f} | {rce1_zoom:.3f} | {rce1_elastic:.3f} | {rce1_quant:.3f} | {rce1_gaussian:.3f} | {rce1_impulse:.3f} | {rce1_shot:.3f} | {rce1_iso:.3f} | {rce1_pixelate:.3f} | {rce1_jpeg:.3f} |
| $\\text{{DES}}_{{2}}$ | {clean2:.3f} | {bright2:.3f}     | {dark2:.3f}     | {fog2:.3f}     | {frost2:.3f}     | {snow2:.3f}     | {contrast2:.3f}     | {defocus2:.3f}     | {glass2:.3f}     | {motion2:.3f}     | {zoom2:.3f}     | {elastic2:.3f}     | {quant2:.3f}     | {gaussian2:.3f}     | {impulse2:.3f}     | {shot2:.3f}     | {iso2:.3f}     | {pixelate2:.3f}     | {jpeg2:.3f}     |
| $\\text{{CE}}_{{2}}$  |      -       | {ce2_bright:.3f}  | {ce2_dark:.3f}  | {ce2_fog:.3f}  | {ce2_frost:.3f}  | {ce2_snow:.3f}  | {ce2_contrast:.3f}  | {ce2_defocus:.3f}  | {ce2_glass:.3f}  | {ce2_motion:.3f}  | {ce2_zoom:.3f}  | {ce2_elastic:.3f}  | {ce2_quant:.3f}  | {ce2_gaussian:.3f}  | {ce2_impulse:.3f}  | {ce2_shot:.3f}  | {ce2_iso:.3f}  | {ce2_pixelate:.3f}  | {ce2_jpeg:.3f}  |
| $\\text{{RCE}}_{{2}}$ |      -       | {rce2_bright:.3f} | {rce2_dark:.3f} | {rce2_fog:.3f} | {rce2_frost:.3f} | {rce2_snow:.3f} | {rce2_contrast:.3f} | {rce2_defocus:.3f} | {rce2_glass:.3f} | {rce2_motion:.3f} | {rce2_zoom:.3f} | {rce2_elastic:.3f} | {rce2_quant:.3f} | {rce2_gaussian:.3f} | {rce2_impulse:.3f} | {rce2_shot:.3f} | {rce2_iso:.3f} | {rce2_pixelate:.3f} | {rce2_jpeg:.3f} |
| $\\text{{DES}}_{{3}}$ | {clean3:.3f} | {bright3:.3f}     | {dark3:.3f}     | {fog3:.3f}     | {frost3:.3f}     | {snow3:.3f}     | {contrast3:.3f}     | {defocus3:.3f}     | {glass3:.3f}     | {motion3:.3f}     | {zoom3:.3f}     | {elastic3:.3f}     | {quant3:.3f}     | {gaussian3:.3f}     | {impulse3:.3f}     | {shot3:.3f}     | {iso3:.3f}     | {pixelate3:.3f}     | {jpeg3:.3f}     |
| $\\text{{CE}}_{{3}}$  |      -       | {ce3_bright:.3f}  | {ce3_dark:.3f}  | {ce3_fog:.3f}  | {ce3_frost:.3f}  | {ce3_snow:.3f}  | {ce3_contrast:.3f}  | {ce3_defocus:.3f}  | {ce3_glass:.3f}  | {ce3_motion:.3f}  | {ce3_zoom:.3f}  | {ce3_elastic:.3f}  | {ce3_quant:.3f}  | {ce3_gaussian:.3f}  | {ce3_impulse:.3f}  | {ce3_shot:.3f}  | {ce3_iso:.3f}  | {ce3_pixelate:.3f}  | {ce3_jpeg:.3f}  |
| $\\text{{RCE}}_{{3}}$ |      -       | {rce3_bright:.3f} | {rce3_dark:.3f} | {rce3_fog:.3f} | {rce3_frost:.3f} | {rce3_snow:.3f} | {rce3_contrast:.3f} | {rce3_defocus:.3f} | {rce3_glass:.3f} | {rce3_motion:.3f} | {rce3_zoom:.3f} | {rce3_elastic:.3f} | {rce3_quant:.3f} | {rce3_gaussian:.3f} | {rce3_impulse:.3f} | {rce3_shot:.3f} | {rce3_iso:.3f} | {rce3_pixelate:.3f} | {rce3_jpeg:.3f} |

- **Summary:** $\\text{{mCE}}_1 =$ {mce1:.3f}, $\\text{{RmCE}}_1 =$ {rmce1:.3f}, $\\text{{mCE}}_2 =$ {mce2:.3f}, $\\text{{RmCE}}_2 =$ {rmce2:.3f}, $\\text{{mCE}}_3 =$ {mce3:.3f}, $\\text{{RmCE}}_3 =$ {rmce3:.3f}

""".format(
    # DES1 = abs_rel - a1 + 1
    clean1 = clean1,
    bright1 = bright1, dark1 = dark1, fog1 = fog1, frost1 = frost1, snow1 = snow1, contrast1 = contrast1,
    defocus1 = defocus1, glass1 = glass1, motion1 = motion1, zoom1 = zoom1, elastic1 = elastic1, quant1 = quant1,
    gaussian1 = gaussian1, impulse1 = impulse1, shot1 = shot1, iso1 = iso1, pixelate1 = pixelate1, jpeg1 = jpeg1,
    # DES2 = 0.5 * (abs_rel - a1 + 1)
    clean2 = clean2,
    bright2 = bright2, dark2 = dark2, fog2 = fog2, frost2 = frost2, snow2 = snow2, contrast2 = contrast2,
    defocus2 = defocus2, glass2 = glass2, motion2 = motion2, zoom2 = zoom2, elastic2 = elastic2, quant2 = quant2,
    gaussian2 = gaussian2, impulse2 = impulse2, shot2 = shot2, iso2 = iso2, pixelate2 = pixelate2, jpeg2 = jpeg2,
    # DES3 = abs_rel / a1
    clean3 = clean3,
    bright3 = bright3, dark3 = dark3, fog3 = fog3, frost3 = frost3, snow3 = snow3, contrast3 = contrast3,
    defocus3 = defocus3, glass3 = glass3, motion3 = motion3, zoom3 = zoom3, elastic3 = elastic3, quant3 = quant3,
    gaussian3 = gaussian3, impulse3 = impulse3, shot3 = shot3, iso3 = iso3, pixelate3 = pixelate3, jpeg3 = jpeg3,
    # CE (DES1)
    ce1_bright = ce1_bright, ce1_dark = ce1_dark, ce1_fog = ce1_fog, ce1_frost = ce1_frost, ce1_snow = ce1_snow, ce1_contrast = ce1_contrast,
    ce1_defocus = ce1_defocus, ce1_glass = ce1_glass, ce1_motion = ce1_motion, ce1_zoom = ce1_zoom, ce1_elastic = ce1_elastic, ce1_quant = ce1_quant,
    ce1_gaussian = ce1_gaussian, ce1_impulse = ce1_impulse, ce1_shot = ce1_shot, ce1_iso = ce1_iso, ce1_pixelate = ce1_pixelate, ce1_jpeg = ce1_jpeg,
    # RCE (DES1)
    rce1_bright = rce1_bright, rce1_dark = rce1_dark, rce1_fog = rce1_fog, rce1_frost = rce1_frost, rce1_snow = rce1_snow, rce1_contrast = rce1_contrast,
    rce1_defocus = rce1_defocus, rce1_glass = rce1_glass, rce1_motion = rce1_motion, rce1_zoom = rce1_zoom, rce1_elastic = rce1_elastic, rce1_quant = rce1_quant,
    rce1_gaussian = rce1_gaussian, rce1_impulse = rce1_impulse, rce1_shot = rce1_shot, rce1_iso = rce1_iso, rce1_pixelate = rce1_pixelate, rce1_jpeg = rce1_jpeg,
    # CE (DES2)
    ce2_bright = ce2_bright, ce2_dark = ce2_dark, ce2_fog = ce2_fog, ce2_frost = ce2_frost, ce2_snow = ce2_snow, ce2_contrast = ce2_contrast,
    ce2_defocus = ce2_defocus, ce2_glass = ce2_glass, ce2_motion = ce2_motion, ce2_zoom = ce2_zoom, ce2_elastic = ce2_elastic, ce2_quant = ce2_quant,
    ce2_gaussian = ce2_gaussian, ce2_impulse = ce2_impulse, ce2_shot = ce2_shot, ce2_iso = ce2_iso, ce2_pixelate = ce2_pixelate, ce2_jpeg = ce2_jpeg,
    # RCE (DES2)
    rce2_bright = rce2_bright, rce2_dark = rce2_dark, rce2_fog = rce2_fog, rce2_frost = rce2_frost, rce2_snow = rce2_snow, rce2_contrast = rce2_contrast,
    rce2_defocus = rce2_defocus, rce2_glass = rce2_glass, rce2_motion = rce2_motion, rce2_zoom = rce2_zoom, rce2_elastic = rce2_elastic, rce2_quant = rce2_quant,
    rce2_gaussian = rce2_gaussian, rce2_impulse = rce2_impulse, rce2_shot = rce2_shot, rce2_iso = rce2_iso, rce2_pixelate = rce2_pixelate, rce2_jpeg = rce2_jpeg,
    # CE (DES3)
    ce3_bright = ce3_bright, ce3_dark = ce3_dark, ce3_fog = ce3_fog, ce3_frost = ce3_frost, ce3_snow = ce3_snow, ce3_contrast = ce3_contrast,
    ce3_defocus = ce3_defocus, ce3_glass = ce3_glass, ce3_motion = ce3_motion, ce3_zoom = ce3_zoom, ce3_elastic = ce3_elastic, ce3_quant = ce3_quant,
    ce3_gaussian = ce3_gaussian, ce3_impulse = ce3_impulse, ce3_shot = ce3_shot, ce3_iso = ce3_iso, ce3_pixelate = ce3_pixelate, ce3_jpeg = ce3_jpeg,
    # RCE (DES3)
    rce3_bright = rce3_bright, rce3_dark = rce3_dark, rce3_fog = rce3_fog, rce3_frost = rce3_frost, rce3_snow = rce3_snow, rce3_contrast = rce3_contrast,
    rce3_defocus = rce3_defocus, rce3_glass = rce3_glass, rce3_motion = rce3_motion, rce3_zoom = rce3_zoom, rce3_elastic = rce3_elastic, rce3_quant = rce3_quant,
    rce3_gaussian = rce3_gaussian, rce3_impulse = rce3_impulse, rce3_shot = rce3_shot, rce3_iso = rce3_iso, rce3_pixelate = rce3_pixelate, rce3_jpeg = rce3_jpeg,
    # mCE & RmCE (DES1)
    mce1 = np.mean(
        [ce1_bright, ce1_dark, ce1_fog, ce1_frost, ce1_snow, ce1_contrast,
        ce1_defocus, ce1_glass, ce1_motion, ce1_zoom, ce1_elastic, ce1_quant,
        ce1_gaussian, ce1_impulse, ce1_shot, ce1_iso, ce1_pixelate, ce1_jpeg,]
    ),
    rmce1 = np.mean(
        [rce1_bright, rce1_dark, rce1_fog, rce1_frost, rce1_snow, rce1_contrast,
        rce1_defocus, rce1_glass, rce1_motion, rce1_zoom, rce1_elastic, rce1_quant,
        rce1_gaussian, rce1_impulse, rce1_shot, rce1_iso, rce1_pixelate, rce1_jpeg,]
    ),
    # mCE & RmCE (DES2)
    mce2 = np.mean(
        [ce2_bright, ce2_dark, ce2_fog, ce2_frost, ce2_snow, ce2_contrast,
        ce2_defocus, ce2_glass, ce2_motion, ce2_zoom, ce2_elastic, ce2_quant,
        ce2_gaussian, ce2_impulse, ce2_shot, ce2_iso, ce2_pixelate, ce2_jpeg,]
    ),
    rmce2 = np.mean(
        [rce2_bright, rce2_dark, rce2_fog, rce2_frost, rce2_snow, rce2_contrast,
        rce2_defocus, rce2_glass, rce2_motion, rce2_zoom, rce2_elastic, rce2_quant,
        rce2_gaussian, rce2_impulse, rce2_shot, rce2_iso, rce2_pixelate, rce2_jpeg,]
    ),
    # mCE & RmCE (DES3)
    mce3 = np.mean(
        [ce3_bright, ce3_dark, ce3_fog, ce3_frost, ce3_snow, ce3_contrast,
        ce3_defocus, ce3_glass, ce3_motion, ce3_zoom, ce3_elastic, ce3_quant,
        ce3_gaussian, ce3_impulse, ce3_shot, ce3_iso, ce3_pixelate, ce3_jpeg,]
    ),
    rmce3 = np.mean(
        [rce3_bright, rce3_dark, rce3_fog, rce3_frost, rce3_snow, rce3_contrast,
        rce3_defocus, rce3_glass, rce3_motion, rce3_zoom, rce3_elastic, rce3_quant,
        rce3_gaussian, rce3_impulse, rce3_shot, rce3_iso, rce3_pixelate, rce3_jpeg,]
    )
)

    return printsummary


def create_corruptions(image, corruption, severity):
    """
    Create corruptions on-the-fly.
    """
    corrupted = corrupt(image, corruption_name=corruption, severity=severity)
    return corrupted

def create_corruption_dark(image, corruption, severity):
    """
    Create corruption 'Dark' on-the-fly.
    """
    corrupted = low_light(image, severity=int(severity-1))
    return corrupted

def create_corruption_color_quant(image, corruption, severity):
    """
    Create corruption 'Color Quantization' on-the-fly.
    """
    image = transforms.ToPILImage()(image)
    corrupted = color_quant(image, severity=int(severity-1))
    return corrupted

def create_corruption_iso_noise(image, corruption, severity):
    """
    Create corruption 'ISO Noise' on-the-fly.
    """
    corrupted = iso_noise(image, severity=int(severity-1))
    return corrupted

def low_light(x, severity):
    c = [0.60, 0.50, 0.40, 0.30, 0.20][severity]
    x = np.array(x) / 255.
    x_scaled = imadjust(x, x.min(), x.max(), 0, c, gamma=2.) * 255
    x_scaled = poisson_gaussian_noise(x_scaled, severity=severity)
    return x_scaled

def imadjust(x, a, b, c, d, gamma=1):
    y = (((x - a) / (b - a)) ** gamma) * (d - c) + c
    return y

def poisson_gaussian_noise(x, severity):
    c_poisson = 10 * [60, 25, 12, 5, 3][severity]
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255
    c_gauss = 0.1 * [.08, .12, 0.18, 0.26, 0.38][severity]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255
    return Image.fromarray(np.uint8(x))

def color_quant(x, severity):
    bits = 5 - severity
    x = PIL.ImageOps.posterize(x, bits)
    return x

def iso_noise(x, severity):
    c_poisson = 25
    x = np.array(x) / 255.
    x = np.clip(np.random.poisson(x * c_poisson) / c_poisson, 0, 1) * 255.
    c_gauss = 0.7 * [.08, .12, 0.18, 0.26, 0.38][severity]
    x = np.array(x) / 255.
    x = np.clip(x + np.random.normal(size=x.shape, scale= c_gauss), 0, 1) * 255.
    return Image.fromarray(np.uint8(x))
