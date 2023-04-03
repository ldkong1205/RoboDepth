import numpy as np
import PIL
from PIL import Image
from torchvision import transforms
from imagecorruptions import corrupt


def get_scores_monodepth2_mono():
    baseline = {
        'DEE1': dict(), 'DEE2': dict(), 'DEE3': dict()
    }

    # DEE1 = abs_rel - a1 + 1
    baseline['DEE1']['clean']               = 0.238
    baseline['DEE1']['brightness']          = 0.259
    baseline['DEE1']['dark']                = 0.561
    baseline['DEE1']['fog']                 = 0.311
    baseline['DEE1']['frost']               = 0.553
    baseline['DEE1']['snow']                = 1.023
    baseline['DEE1']['contrast']            = 0.373
    baseline['DEE1']['defocus_blur']        = 0.487
    baseline['DEE1']['glass_blur']          = 0.484
    baseline['DEE1']['motion_blur']         = 0.433	
    baseline['DEE1']['zoom_blur']           = 0.402
    baseline['DEE1']['elastic_transform']   = 0.258
    baseline['DEE1']['color_quant']         = 0.386
    baseline['DEE1']['gaussian_noise']      = 0.768
    baseline['DEE1']['impulse_noise']       = 0.779
    baseline['DEE1']['shot_noise']          = 0.681	
    baseline['DEE1']['iso_noise']           = 0.776
    baseline['DEE1']['pixelate']            = 0.289
    baseline['DEE1']['jpeg_compression']    = 0.391

    # DEE2 = 0.5 * (abs_rel - a1 + 1)
    baseline['DEE2']['clean']               = 0.119
    baseline['DEE2']['brightness']          = 0.130
    baseline['DEE2']['dark']                = 0.280
    baseline['DEE2']['fog']                 = 0.155
    baseline['DEE2']['frost']               = 0.277
    baseline['DEE2']['snow']                = 0.511
    baseline['DEE2']['contrast']            = 0.187
    baseline['DEE2']['defocus_blur']        = 0.244	
    baseline['DEE2']['glass_blur']          = 0.242
    baseline['DEE2']['motion_blur']         = 0.216
    baseline['DEE2']['zoom_blur']           = 0.201	
    baseline['DEE2']['elastic_transform']   = 0.129
    baseline['DEE2']['color_quant']         = 0.193
    baseline['DEE2']['gaussian_noise']      = 0.384
    baseline['DEE2']['impulse_noise']       = 0.389
    baseline['DEE2']['shot_noise']          = 0.340
    baseline['DEE2']['iso_noise']           = 0.388
    baseline['DEE2']['pixelate']            = 0.145
    baseline['DEE2']['jpeg_compression']    = 0.196

    # DEE3 = abs_rel / a1
    baseline['DEE3']['clean']               = 0.131
    baseline['DEE3']['brightness']          = 0.140
    baseline['DEE3']['dark']                = 0.330
    baseline['DEE3']['fog']                 = 0.166
    baseline['DEE3']['frost']               = 0.327
    baseline['DEE3']['snow']                = 1.058
    baseline['DEE3']['contrast']            = 0.199
    baseline['DEE3']['defocus_blur']        = 0.271
    baseline['DEE3']['glass_blur']          = 0.270
    baseline['DEE3']['motion_blur']         = 0.236
    baseline['DEE3']['zoom_blur']           = 0.221
    baseline['DEE3']['elastic_transform']   = 0.141
    baseline['DEE3']['color_quant']         = 0.209
    baseline['DEE3']['gaussian_noise']      = 0.551
    baseline['DEE3']['impulse_noise']       = 0.563
    baseline['DEE3']['shot_noise']          = 0.442
    baseline['DEE3']['iso_noise']           = 0.557
    baseline['DEE3']['pixelate']            = 0.158
    baseline['DEE3']['jpeg_compression']    = 0.215

    return baseline


def get_scores_monodepth2_stereo():
    baseline = {
        'DEE1': dict(), 'DEE2': dict(), 'DEE3': dict()
    }

    # DEE1 = abs_rel - a1 + 1
    baseline['DEE1']['clean']               = 0.246
    baseline['DEE1']['brightness']          = 0.266
    baseline['DEE1']['dark']                = 0.696
    baseline['DEE1']['fog']                 = 0.322
    baseline['DEE1']['frost']               = 0.610
    baseline['DEE1']['snow']                = 1.031
    baseline['DEE1']['contrast']            = 0.468
    baseline['DEE1']['defocus_blur']        = 0.779
    baseline['DEE1']['glass_blur']          = 0.663
    baseline['DEE1']['motion_blur']         = 0.529	
    baseline['DEE1']['zoom_blur']           = 0.419
    baseline['DEE1']['elastic_transform']   = 0.270
    baseline['DEE1']['color_quant']         = 0.399
    baseline['DEE1']['gaussian_noise']      = 0.985
    baseline['DEE1']['impulse_noise']       = 1.018
    baseline['DEE1']['shot_noise']          = 0.927	
    baseline['DEE1']['iso_noise']           = 0.985	
    baseline['DEE1']['pixelate']            = 0.288
    baseline['DEE1']['jpeg_compression']    = 0.387

    # DEE2 = 0.5 * (abs_rel - a1 + 1)
    baseline['DEE2']['clean']               = 0.123
    baseline['DEE2']['brightness']          = 0.133
    baseline['DEE2']['dark']                = 0.348
    baseline['DEE2']['fog']                 = 0.161
    baseline['DEE2']['frost']               = 0.305
    baseline['DEE2']['snow']                = 0.515
    baseline['DEE2']['contrast']            = 0.234
    baseline['DEE2']['defocus_blur']        = 0.390	
    baseline['DEE2']['glass_blur']          = 0.332
    baseline['DEE2']['motion_blur']         = 0.264
    baseline['DEE2']['zoom_blur']           = 0.209
    baseline['DEE2']['elastic_transform']   = 0.135
    baseline['DEE2']['color_quant']         = 0.200
    baseline['DEE2']['gaussian_noise']      = 0.492
    baseline['DEE2']['impulse_noise']       = 0.509
    baseline['DEE2']['shot_noise']          = 0.463
    baseline['DEE2']['iso_noise']           = 0.493
    baseline['DEE2']['pixelate']            = 0.144
    baseline['DEE2']['jpeg_compression']    = 0.194

    # DEE3 = abs_rel / a1
    baseline['DEE3']['clean']               = 0.127
    baseline['DEE3']['brightness']          = 0.137
    baseline['DEE3']['dark']                = 0.455
    baseline['DEE3']['fog']                 = 0.163
    baseline['DEE3']['frost']               = 0.373
    baseline['DEE3']['snow']                = 1.085
    baseline['DEE3']['contrast']            = 0.256
    baseline['DEE3']['defocus_blur']        = 0.567
    baseline['DEE3']['glass_blur']          = 0.418
    baseline['DEE3']['motion_blur']         = 0.298
    baseline['DEE3']['zoom_blur']           = 0.222
    baseline['DEE3']['elastic_transform']   = 0.138
    baseline['DEE3']['color_quant']         = 0.205
    baseline['DEE3']['gaussian_noise']      = 0.960
    baseline['DEE3']['impulse_noise']       = 1.052
    baseline['DEE3']['shot_noise']          = 0.824
    baseline['DEE3']['iso_noise']           = 0.960
    baseline['DEE3']['pixelate']            = 0.151
    baseline['DEE3']['jpeg_compression']    = 0.202

    return baseline


def get_scores_monodepth2_mono_stereo():
    baseline = {
        'DEE1': dict(), 'DEE2': dict(), 'DEE3': dict()
    }

    # DEE1 = abs_rel - a1 + 1
    baseline['DEE1']['clean']               = 0.232
    baseline['DEE1']['brightness']          = 0.255
    baseline['DEE1']['dark']                = 0.808
    baseline['DEE1']['fog']                 = 0.300
    baseline['DEE1']['frost']               = 0.589
    baseline['DEE1']['snow']                = 1.073
    baseline['DEE1']['contrast']            = 0.398
    baseline['DEE1']['defocus_blur']        = 0.895
    baseline['DEE1']['glass_blur']          = 0.692
    baseline['DEE1']['motion_blur']         = 0.565	
    baseline['DEE1']['zoom_blur']           = 0.408
    baseline['DEE1']['elastic_transform']   = 0.256
    baseline['DEE1']['color_quant']         = 0.407
    baseline['DEE1']['gaussian_noise']      = 1.154
    baseline['DEE1']['impulse_noise']       = 1.209
    baseline['DEE1']['shot_noise']          = 1.121
    baseline['DEE1']['iso_noise']           = 1.258
    baseline['DEE1']['pixelate']            = 0.271
    baseline['DEE1']['jpeg_compression']    = 0.358

    # DEE2 = 0.5 * (abs_rel - a1 + 1)
    baseline['DEE2']['clean']               = 0.116
    baseline['DEE2']['brightness']          = 0.127
    baseline['DEE2']['dark']                = 0.404
    baseline['DEE2']['fog']                 = 0.150
    baseline['DEE2']['frost']               = 0.295
    baseline['DEE2']['snow']                = 0.536
    baseline['DEE2']['contrast']            = 0.199
    baseline['DEE2']['defocus_blur']        = 0.447
    baseline['DEE2']['glass_blur']          = 0.346
    baseline['DEE2']['motion_blur']         = 0.283
    baseline['DEE2']['zoom_blur']           = 0.204
    baseline['DEE2']['elastic_transform']   = 0.128
    baseline['DEE2']['color_quant']         = 0.203
    baseline['DEE2']['gaussian_noise']      = 0.577
    baseline['DEE2']['impulse_noise']       = 0.605
    baseline['DEE2']['shot_noise']          = 0.560
    baseline['DEE2']['iso_noise']           = 0.629
    baseline['DEE2']['pixelate']            = 0.136
    baseline['DEE2']['jpeg_compression']    = 0.179

    # DEE3 = abs_rel / a1
    baseline['DEE3']['clean']               = 0.121
    baseline['DEE3']['brightness']          = 0.131
    baseline['DEE3']['dark']                = 0.607
    baseline['DEE3']['fog']                 = 0.153
    baseline['DEE3']['frost']               = 0.348
    baseline['DEE3']['snow']                = 1.214
    baseline['DEE3']['contrast']            = 0.205
    baseline['DEE3']['defocus_blur']        = 0.755
    baseline['DEE3']['glass_blur']          = 0.446
    baseline['DEE3']['motion_blur']         = 0.325
    baseline['DEE3']['zoom_blur']           = 0.216
    baseline['DEE3']['elastic_transform']   = 0.131
    baseline['DEE3']['color_quant']         = 0.210
    baseline['DEE3']['gaussian_noise']      = 1.544
    baseline['DEE3']['impulse_noise']       = 1.872
    baseline['DEE3']['shot_noise']          = 1.398
    baseline['DEE3']['iso_noise']           = 2.293
    baseline['DEE3']['pixelate']            = 0.142
    baseline['DEE3']['jpeg_compression']    = 0.184

    return baseline


def logger_summary(info, summary, baseline):

    # DEE1 = abs_rel - a1 + 1
    clean1    = info['clean']['DEE1']                      # 00
    bright1   = summary['avgs_DEE1']['brightness']         # 01
    dark1     = summary['avgs_DEE1']['dark']               # 02
    fog1      = summary['avgs_DEE1']['fog']                # 03
    frost1    = summary['avgs_DEE1']['frost']              # 04
    snow1     = summary['avgs_DEE1']['snow']               # 05
    contrast1 = summary['avgs_DEE1']['contrast']           # 06
    defocus1  = summary['avgs_DEE1']['defocus_blur']       # 07
    glass1    = summary['avgs_DEE1']['glass_blur']         # 08
    motion1   = summary['avgs_DEE1']['motion_blur']        # 09
    zoom1     = summary['avgs_DEE1']['zoom_blur']          # 10
    elastic1  = summary['avgs_DEE1']['elastic_transform']  # 11
    quant1    = summary['avgs_DEE1']['color_quant']        # 12
    gaussian1 = summary['avgs_DEE1']['gaussian_noise']     # 13
    impulse1  = summary['avgs_DEE1']['impulse_noise']      # 14
    shot1     = summary['avgs_DEE1']['shot_noise']         # 15
    iso1      = summary['avgs_DEE1']['iso_noise']          # 16
    pixelate1 = summary['avgs_DEE1']['pixelate']           # 17
    jpeg1     = summary['avgs_DEE1']['jpeg_compression']   # 18

    # DEE2 = 0.5 * (abs_rel - a1 + 1)
    clean2    = info['clean']['DEE2']                      # 00 -
    bright2   = summary['avgs_DEE2']['brightness']         # 01
    dark2     = summary['avgs_DEE2']['dark']               # 02
    fog2      = summary['avgs_DEE2']['fog']                # 03
    frost2    = summary['avgs_DEE2']['frost']              # 04
    snow2     = summary['avgs_DEE2']['snow']               # 05
    contrast2 = summary['avgs_DEE2']['contrast']           # 06
    defocus2  = summary['avgs_DEE2']['defocus_blur']       # 07
    glass2    = summary['avgs_DEE2']['glass_blur']         # 08
    motion2   = summary['avgs_DEE2']['motion_blur']        # 09
    zoom2     = summary['avgs_DEE2']['zoom_blur']          # 10
    elastic2  = summary['avgs_DEE2']['elastic_transform']  # 11
    quant2    = summary['avgs_DEE2']['color_quant']        # 12
    gaussian2 = summary['avgs_DEE2']['gaussian_noise']     # 13
    impulse2  = summary['avgs_DEE2']['impulse_noise']      # 14
    shot2     = summary['avgs_DEE2']['shot_noise']         # 15
    iso2      = summary['avgs_DEE2']['iso_noise']          # 16
    pixelate2 = summary['avgs_DEE2']['pixelate']           # 17
    jpeg2     = summary['avgs_DEE2']['jpeg_compression']   # 18

    # DEE3 = abs_rel / a1
    clean3    = info['clean']['DEE3']                      # 00
    bright3   = summary['avgs_DEE3']['brightness']         # 01
    dark3     = summary['avgs_DEE3']['dark']               # 02
    fog3      = summary['avgs_DEE3']['fog']                # 03
    frost3    = summary['avgs_DEE3']['frost']              # 04
    snow3     = summary['avgs_DEE3']['snow']               # 05
    contrast3 = summary['avgs_DEE3']['contrast']           # 06
    defocus3  = summary['avgs_DEE3']['defocus_blur']       # 07
    glass3    = summary['avgs_DEE3']['glass_blur']         # 08
    motion3   = summary['avgs_DEE3']['motion_blur']        # 09
    zoom3     = summary['avgs_DEE3']['zoom_blur']          # 10
    elastic3  = summary['avgs_DEE3']['elastic_transform']  # 11
    quant3    = summary['avgs_DEE3']['color_quant']        # 12
    gaussian3 = summary['avgs_DEE3']['gaussian_noise']     # 13
    impulse3  = summary['avgs_DEE3']['impulse_noise']      # 14
    shot3     = summary['avgs_DEE3']['shot_noise']         # 15
    iso3      = summary['avgs_DEE3']['iso_noise']          # 16
    pixelate3 = summary['avgs_DEE3']['pixelate']           # 17
    jpeg3     = summary['avgs_DEE3']['jpeg_compression']   # 18

    # CE (DEE1)
    ce1_bright   = bright1   / baseline['DEE1']['brightness']
    ce1_dark     = dark1     / baseline['DEE1']['dark']
    ce1_fog      = fog1      / baseline['DEE1']['fog']
    ce1_frost    = frost1    / baseline['DEE1']['frost']
    ce1_snow     = snow1     / baseline['DEE1']['snow']
    ce1_contrast = contrast1 / baseline['DEE1']['contrast']
    ce1_defocus  = defocus1  / baseline['DEE1']['defocus_blur']
    ce1_glass    = glass1    / baseline['DEE1']['glass_blur']
    ce1_motion   = motion1   / baseline['DEE1']['motion_blur']
    ce1_zoom     = zoom1     / baseline['DEE1']['zoom_blur']
    ce1_elastic  = elastic1  / baseline['DEE1']['elastic_transform']
    ce1_quant    = quant1    / baseline['DEE1']['color_quant']
    ce1_gaussian = gaussian1 / baseline['DEE1']['gaussian_noise']
    ce1_impulse  = impulse1  / baseline['DEE1']['impulse_noise']
    ce1_shot     = shot1     / baseline['DEE1']['shot_noise']
    ce1_iso      = iso1      / baseline['DEE1']['iso_noise']
    ce1_pixelate = pixelate1 / baseline['DEE1']['pixelate']
    ce1_jpeg     = jpeg1     / baseline['DEE1']['jpeg_compression']

    # CE (DEE2)
    ce2_bright   = bright2   / baseline['DEE2']['brightness']
    ce2_dark     = dark2     / baseline['DEE2']['dark']
    ce2_fog      = fog2      / baseline['DEE2']['fog']
    ce2_frost    = frost2    / baseline['DEE2']['frost']
    ce2_snow     = snow2     / baseline['DEE2']['snow']
    ce2_contrast = contrast2 / baseline['DEE2']['contrast']
    ce2_defocus  = defocus2  / baseline['DEE2']['defocus_blur']
    ce2_glass    = glass2    / baseline['DEE2']['glass_blur']
    ce2_motion   = motion2   / baseline['DEE2']['motion_blur']
    ce2_zoom     = zoom2     / baseline['DEE2']['zoom_blur']
    ce2_elastic  = elastic2  / baseline['DEE2']['elastic_transform']
    ce2_quant    = quant2    / baseline['DEE2']['color_quant']
    ce2_gaussian = gaussian2 / baseline['DEE2']['gaussian_noise']
    ce2_impulse  = impulse2  / baseline['DEE2']['impulse_noise']
    ce2_shot     = shot2     / baseline['DEE2']['shot_noise']
    ce2_iso      = iso2      / baseline['DEE2']['iso_noise']
    ce2_pixelate = pixelate2 / baseline['DEE2']['pixelate']
    ce2_jpeg     = jpeg2     / baseline['DEE2']['jpeg_compression']

    # CE (DEE3)
    ce3_bright   = bright3   / baseline['DEE3']['brightness']
    ce3_dark     = dark3     / baseline['DEE3']['dark']
    ce3_fog      = fog3      / baseline['DEE3']['fog']
    ce3_frost    = frost3    / baseline['DEE3']['frost']
    ce3_snow     = snow3     / baseline['DEE3']['snow']
    ce3_contrast = contrast3 / baseline['DEE3']['contrast']
    ce3_defocus  = defocus3  / baseline['DEE3']['defocus_blur']
    ce3_glass    = glass3    / baseline['DEE3']['glass_blur']
    ce3_motion   = motion3   / baseline['DEE3']['motion_blur']
    ce3_zoom     = zoom3     / baseline['DEE3']['zoom_blur']
    ce3_elastic  = elastic3  / baseline['DEE3']['elastic_transform']
    ce3_quant    = quant3    / baseline['DEE3']['color_quant']
    ce3_gaussian = gaussian3 / baseline['DEE3']['gaussian_noise']
    ce3_impulse  = impulse3  / baseline['DEE3']['impulse_noise']
    ce3_shot     = shot3     / baseline['DEE3']['shot_noise']
    ce3_iso      = iso3      / baseline['DEE3']['iso_noise']
    ce3_pixelate = pixelate3 / baseline['DEE3']['pixelate']
    ce3_jpeg     = jpeg3     / baseline['DEE3']['jpeg_compression']

    # RCE (DEE1)
    rce1_bright    = (clean1 - bright1)   / (baseline['DEE1']['clean'] - baseline['DEE1']['brightness'])
    rce1_dark      = (clean1 - dark1)     / (baseline['DEE1']['clean'] - baseline['DEE1']['dark'])
    rce1_fog       = (clean1 - fog1)      / (baseline['DEE1']['clean'] - baseline['DEE1']['fog'])
    rce1_frost     = (clean1 - frost1)    / (baseline['DEE1']['clean'] - baseline['DEE1']['frost'])
    rce1_snow      = (clean1 - snow1)     / (baseline['DEE1']['clean'] - baseline['DEE1']['snow'])
    rce1_contrast  = (clean1 - contrast1) / (baseline['DEE1']['clean'] - baseline['DEE1']['contrast'])
    rce1_defocus   = (clean1 - defocus1)  / (baseline['DEE1']['clean'] - baseline['DEE1']['defocus_blur'])
    rce1_glass     = (clean1 - glass1)    / (baseline['DEE1']['clean'] - baseline['DEE1']['glass_blur'])
    rce1_motion    = (clean1 - motion1)   / (baseline['DEE1']['clean'] - baseline['DEE1']['motion_blur'])
    rce1_zoom      = (clean1 - zoom1)     / (baseline['DEE1']['clean'] - baseline['DEE1']['zoom_blur'])
    rce1_elastic   = (clean1 - elastic1)  / (baseline['DEE1']['clean'] - baseline['DEE1']['elastic_transform'])
    rce1_quant     = (clean1 - quant1)    / (baseline['DEE1']['clean'] - baseline['DEE1']['color_quant'])
    rce1_gaussian  = (clean1 - gaussian1) / (baseline['DEE1']['clean'] - baseline['DEE1']['gaussian_noise'])
    rce1_impulse   = (clean1 - impulse1)  / (baseline['DEE1']['clean'] - baseline['DEE1']['impulse_noise'])
    rce1_shot      = (clean1 - shot1)     / (baseline['DEE1']['clean'] - baseline['DEE1']['shot_noise'])
    rce1_iso       = (clean1 - iso1)      / (baseline['DEE1']['clean'] - baseline['DEE1']['iso_noise'])
    rce1_pixelate  = (clean1 - pixelate1) / (baseline['DEE1']['clean'] - baseline['DEE1']['pixelate'])
    rce1_jpeg      = (clean1 - jpeg1)     / (baseline['DEE1']['clean'] - baseline['DEE1']['jpeg_compression'])

    # RCE (DEE2)
    rce2_bright    = (clean2 - bright2)   / (baseline['DEE2']['clean'] - baseline['DEE2']['brightness'])
    rce2_dark      = (clean2 - dark2)     / (baseline['DEE2']['clean'] - baseline['DEE2']['dark'])
    rce2_fog       = (clean2 - fog2)      / (baseline['DEE2']['clean'] - baseline['DEE2']['fog'])
    rce2_frost     = (clean2 - frost2)    / (baseline['DEE2']['clean'] - baseline['DEE2']['frost'])
    rce2_snow      = (clean2 - snow2)     / (baseline['DEE2']['clean'] - baseline['DEE2']['snow'])
    rce2_contrast  = (clean2 - contrast2) / (baseline['DEE2']['clean'] - baseline['DEE2']['contrast'])
    rce2_defocus   = (clean2 - defocus2)  / (baseline['DEE2']['clean'] - baseline['DEE2']['defocus_blur'])
    rce2_glass     = (clean2 - glass2)    / (baseline['DEE2']['clean'] - baseline['DEE2']['glass_blur'])
    rce2_motion    = (clean2 - motion2)   / (baseline['DEE2']['clean'] - baseline['DEE2']['motion_blur'])
    rce2_zoom      = (clean2 - zoom2)     / (baseline['DEE2']['clean'] - baseline['DEE2']['zoom_blur'])
    rce2_elastic   = (clean2 - elastic2)  / (baseline['DEE2']['clean'] - baseline['DEE2']['elastic_transform'])
    rce2_quant     = (clean2 - quant2)    / (baseline['DEE2']['clean'] - baseline['DEE2']['color_quant'])
    rce2_gaussian  = (clean2 - gaussian2) / (baseline['DEE2']['clean'] - baseline['DEE2']['gaussian_noise'])
    rce2_impulse   = (clean2 - impulse2)  / (baseline['DEE2']['clean'] - baseline['DEE2']['impulse_noise'])
    rce2_shot      = (clean2 - shot2)     / (baseline['DEE2']['clean'] - baseline['DEE2']['shot_noise'])
    rce2_iso       = (clean2 - iso2)      / (baseline['DEE2']['clean'] - baseline['DEE2']['iso_noise'])
    rce2_pixelate  = (clean2 - pixelate2) / (baseline['DEE2']['clean'] - baseline['DEE2']['pixelate'])
    rce2_jpeg      = (clean2 - jpeg2)     / (baseline['DEE2']['clean'] - baseline['DEE2']['jpeg_compression'])

    # RCE (DEE3)
    rce3_bright    = (clean3 - bright3)   / (baseline['DEE3']['clean'] - baseline['DEE3']['brightness'])
    rce3_dark      = (clean3 - dark3)     / (baseline['DEE3']['clean'] - baseline['DEE3']['dark'])
    rce3_fog       = (clean3 - fog3)      / (baseline['DEE3']['clean'] - baseline['DEE3']['fog'])
    rce3_frost     = (clean3 - frost3)    / (baseline['DEE3']['clean'] - baseline['DEE3']['frost'])
    rce3_snow      = (clean3 - snow3)     / (baseline['DEE3']['clean'] - baseline['DEE3']['snow'])
    rce3_contrast  = (clean3 - contrast3) / (baseline['DEE3']['clean'] - baseline['DEE3']['contrast'])
    rce3_defocus   = (clean3 - defocus3)  / (baseline['DEE3']['clean'] - baseline['DEE3']['defocus_blur'])
    rce3_glass     = (clean3 - glass3)    / (baseline['DEE3']['clean'] - baseline['DEE3']['glass_blur'])
    rce3_motion    = (clean3 - motion3)   / (baseline['DEE3']['clean'] - baseline['DEE3']['motion_blur'])
    rce3_zoom      = (clean3 - zoom3)     / (baseline['DEE3']['clean'] - baseline['DEE3']['zoom_blur'])
    rce3_elastic   = (clean3 - elastic3)  / (baseline['DEE3']['clean'] - baseline['DEE3']['elastic_transform'])
    rce3_quant     = (clean3 - quant3)    / (baseline['DEE3']['clean'] - baseline['DEE3']['color_quant'])
    rce3_gaussian  = (clean3 - gaussian3) / (baseline['DEE3']['clean'] - baseline['DEE3']['gaussian_noise'])
    rce3_impulse   = (clean3 - impulse3)  / (baseline['DEE3']['clean'] - baseline['DEE3']['impulse_noise'])
    rce3_shot      = (clean3 - shot3)     / (baseline['DEE3']['clean'] - baseline['DEE3']['shot_noise'])
    rce3_iso       = (clean3 - iso3)      / (baseline['DEE3']['clean'] - baseline['DEE3']['iso_noise'])
    rce3_pixelate  = (clean3 - pixelate3) / (baseline['DEE3']['clean'] - baseline['DEE3']['pixelate'])
    rce3_jpeg      = (clean3 - jpeg3)     / (baseline['DEE3']['clean'] - baseline['DEE3']['jpeg_compression'])

    # RR (DEE1)
    rr1_bright    = (1 - bright1)   / (1 - clean1)
    rr1_dark      = (1 - dark1)     / (1 - clean1)
    rr1_fog       = (1 - fog1)      / (1 - clean1)
    rr1_frost     = (1 - frost1)    / (1 - clean1)
    rr1_snow      = (1 - snow1)     / (1 - clean1)
    rr1_contrast  = (1 - contrast1) / (1 - clean1)
    rr1_defocus   = (1 - defocus1)  / (1 - clean1)
    rr1_glass     = (1 - glass1)    / (1 - clean1)
    rr1_motion    = (1 - motion1)   / (1 - clean1)
    rr1_zoom      = (1 - zoom1)     / (1 - clean1)
    rr1_elastic   = (1 - elastic1)  / (1 - clean1)
    rr1_quant     = (1 - quant1)    / (1 - clean1)
    rr1_gaussian  = (1 - gaussian1) / (1 - clean1)
    rr1_impulse   = (1 - impulse1)  / (1 - clean1)
    rr1_shot      = (1 - shot1)     / (1 - clean1)
    rr1_iso       = (1 - iso1)      / (1 - clean1)
    rr1_pixelate  = (1 - pixelate1) / (1 - clean1)
    rr1_jpeg      = (1 - jpeg1)     / (1 - clean1)

    # RR (DEE2)
    rr2_bright    = (1 - bright2)   / (1 - clean2)
    rr2_dark      = (1 - dark2)     / (1 - clean2)
    rr2_fog       = (1 - fog2)      / (1 - clean2)
    rr2_frost     = (1 - frost2)    / (1 - clean2)
    rr2_snow      = (1 - snow2)     / (1 - clean2)
    rr2_contrast  = (1 - contrast2) / (1 - clean2)
    rr2_defocus   = (1 - defocus2)  / (1 - clean2)
    rr2_glass     = (1 - glass2)    / (1 - clean2)
    rr2_motion    = (1 - motion2)   / (1 - clean2)
    rr2_zoom      = (1 - zoom2)     / (1 - clean2)
    rr2_elastic   = (1 - elastic2)  / (1 - clean2)
    rr2_quant     = (1 - quant2)    / (1 - clean2)
    rr2_gaussian  = (1 - gaussian2) / (1 - clean2)
    rr2_impulse   = (1 - impulse2)  / (1 - clean2)
    rr2_shot      = (1 - shot2)     / (1 - clean2)
    rr2_iso       = (1 - iso2)      / (1 - clean2)
    rr2_pixelate  = (1 - pixelate2) / (1 - clean2)
    rr2_jpeg      = (1 - jpeg2)     / (1 - clean2)

    # RR (DEE3)
    rr3_bright    = (1 - bright3)   / (1 - clean3)
    rr3_dark      = (1 - dark3)     / (1 - clean3)
    rr3_fog       = (1 - fog3)      / (1 - clean3)
    rr3_frost     = (1 - frost3)    / (1 - clean3)
    rr3_snow      = (1 - snow3)     / (1 - clean3)
    rr3_contrast  = (1 - contrast3) / (1 - clean3)
    rr3_defocus   = (1 - defocus3)  / (1 - clean3)
    rr3_glass     = (1 - glass3)    / (1 - clean3)
    rr3_motion    = (1 - motion3)   / (1 - clean3)
    rr3_zoom      = (1 - zoom3)     / (1 - clean3)
    rr3_elastic   = (1 - elastic3)  / (1 - clean3)
    rr3_quant     = (1 - quant3)    / (1 - clean3)
    rr3_gaussian  = (1 - gaussian3) / (1 - clean3)
    rr3_impulse   = (1 - impulse3)  / (1 - clean3)
    rr3_shot      = (1 - shot3)     / (1 - clean3)
    rr3_iso       = (1 - iso3)      / (1 - clean3)
    rr3_pixelate  = (1 - pixelate3) / (1 - clean3)
    rr3_jpeg      = (1 - jpeg3)     / (1 - clean3)

    # summary
    printsummary = \
"""\n
###
| Corruption | Clean | Bright | Dark | Fog | Frost | Snow | Contrast | Defocus | Glass | Motion | Zoom | Elastic| Quant| Gaussian | Impulse | Shot | ISO | Pixelate | JPEG | 
| :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: | :--: |          
| $\\text{{DEE}}_{{1}}$ | {clean1:.4f} | {bright1:.4f}     | {dark1:.4f}     | {fog1:.4f}     | {frost1:.4f}     | {snow1:.4f}     | {contrast1:.4f}     | {defocus1:.4f}     | {glass1:.4f}     | {motion1:.4f}     | {zoom1:.4f}     | {elastic1:.4f}     | {quant1:.4f}     | {gaussian1:.4f}     | {impulse1:.4f}     | {shot1:.4f}     | {iso1:.4f}     | {pixelate1:.4f}     | {jpeg1:.4f}     |
| $\\text{{CE}}_{{1}}$  |      -       | {ce1_bright:.4f}  | {ce1_dark:.4f}  | {ce1_fog:.4f}  | {ce1_frost:.4f}  | {ce1_snow:.4f}  | {ce1_contrast:.4f}  | {ce1_defocus:.4f}  | {ce1_glass:.4f}  | {ce1_motion:.4f}  | {ce1_zoom:.4f}  | {ce1_elastic:.4f}  | {ce1_quant:.4f}  | {ce1_gaussian:.4f}  | {ce1_impulse:.4f}  | {ce1_shot:.4f}  | {ce1_iso:.4f}  | {ce1_pixelate:.4f}  | {ce1_jpeg:.4f}  |
| $\\text{{RCE}}_{{1}}$ |      -       | {rce1_bright:.4f} | {rce1_dark:.4f} | {rce1_fog:.4f} | {rce1_frost:.4f} | {rce1_snow:.4f} | {rce1_contrast:.4f} | {rce1_defocus:.4f} | {rce1_glass:.4f} | {rce1_motion:.4f} | {rce1_zoom:.4f} | {rce1_elastic:.4f} | {rce1_quant:.4f} | {rce1_gaussian:.4f} | {rce1_impulse:.4f} | {rce1_shot:.4f} | {rce1_iso:.4f} | {rce1_pixelate:.4f} | {rce1_jpeg:.4f} |
| $\\text{{RR}}_{{1}}$  |      -       | {rr1_bright:.4f}  | {rr1_dark:.4f}  | {rr1_fog:.4f}  | {rr1_frost:.4f}  | {rr1_snow:.4f}  | {rr1_contrast:.4f}  | {rr1_defocus:.4f}  | {rr1_glass:.4f}  | {rr1_motion:.4f}  | {rr1_zoom:.4f}  | {rr1_elastic:.4f}  | {rr1_quant:.4f}  | {rr1_gaussian:.4f}  | {rr1_impulse:.4f}  | {rr1_shot:.4f}  | {rr1_iso:.4f}  | {rr1_pixelate:.4f}  | {rr1_jpeg:.4f}  |
| $\\text{{DEE}}_{{2}}$ | {clean2:.4f} | {bright2:.4f}     | {dark2:.4f}     | {fog2:.4f}     | {frost2:.4f}     | {snow2:.4f}     | {contrast2:.4f}     | {defocus2:.4f}     | {glass2:.4f}     | {motion2:.4f}     | {zoom2:.4f}     | {elastic2:.4f}     | {quant2:.4f}     | {gaussian2:.4f}     | {impulse2:.4f}     | {shot2:.4f}     | {iso2:.4f}     | {pixelate2:.4f}     | {jpeg2:.4f}     |
| $\\text{{CE}}_{{2}}$  |      -       | {ce2_bright:.4f}  | {ce2_dark:.4f}  | {ce2_fog:.4f}  | {ce2_frost:.4f}  | {ce2_snow:.4f}  | {ce2_contrast:.4f}  | {ce2_defocus:.4f}  | {ce2_glass:.4f}  | {ce2_motion:.4f}  | {ce2_zoom:.4f}  | {ce2_elastic:.4f}  | {ce2_quant:.4f}  | {ce2_gaussian:.4f}  | {ce2_impulse:.4f}  | {ce2_shot:.4f}  | {ce2_iso:.4f}  | {ce2_pixelate:.4f}  | {ce2_jpeg:.4f}  |
| $\\text{{RCE}}_{{2}}$ |      -       | {rce2_bright:.4f} | {rce2_dark:.4f} | {rce2_fog:.4f} | {rce2_frost:.4f} | {rce2_snow:.4f} | {rce2_contrast:.4f} | {rce2_defocus:.4f} | {rce2_glass:.4f} | {rce2_motion:.4f} | {rce2_zoom:.4f} | {rce2_elastic:.4f} | {rce2_quant:.4f} | {rce2_gaussian:.4f} | {rce2_impulse:.4f} | {rce2_shot:.4f} | {rce2_iso:.4f} | {rce2_pixelate:.4f} | {rce2_jpeg:.4f} |
| $\\text{{RR}}_{{2}}$  |      -       | {rr2_bright:.4f}  | {rr2_dark:.4f}  | {rr2_fog:.4f}  | {rr2_frost:.4f}  | {rr2_snow:.4f}  | {rr2_contrast:.4f}  | {rr2_defocus:.4f}  | {rr2_glass:.4f}  | {rr2_motion:.4f}  | {rr2_zoom:.4f}  | {rr2_elastic:.4f}  | {rr2_quant:.4f}  | {rr2_gaussian:.4f}  | {rr2_impulse:.4f}  | {rr2_shot:.4f}  | {rr2_iso:.4f}  | {rr2_pixelate:.4f}  | {rr2_jpeg:.4f}  |
| $\\text{{DEE}}_{{3}}$ | {clean3:.4f} | {bright3:.4f}     | {dark3:.4f}     | {fog3:.4f}     | {frost3:.4f}     | {snow3:.4f}     | {contrast3:.4f}     | {defocus3:.4f}     | {glass3:.4f}     | {motion3:.4f}     | {zoom3:.4f}     | {elastic3:.4f}     | {quant3:.4f}     | {gaussian3:.4f}     | {impulse3:.4f}     | {shot3:.4f}     | {iso3:.4f}     | {pixelate3:.4f}     | {jpeg3:.4f}     |
| $\\text{{CE}}_{{3}}$  |      -       | {ce3_bright:.4f}  | {ce3_dark:.4f}  | {ce3_fog:.4f}  | {ce3_frost:.4f}  | {ce3_snow:.4f}  | {ce3_contrast:.4f}  | {ce3_defocus:.4f}  | {ce3_glass:.4f}  | {ce3_motion:.4f}  | {ce3_zoom:.4f}  | {ce3_elastic:.4f}  | {ce3_quant:.4f}  | {ce3_gaussian:.4f}  | {ce3_impulse:.4f}  | {ce3_shot:.4f}  | {ce3_iso:.4f}  | {ce3_pixelate:.4f}  | {ce3_jpeg:.4f}  |
| $\\text{{RCE}}_{{3}}$ |      -       | {rce3_bright:.4f} | {rce3_dark:.4f} | {rce3_fog:.4f} | {rce3_frost:.4f} | {rce3_snow:.4f} | {rce3_contrast:.4f} | {rce3_defocus:.4f} | {rce3_glass:.4f} | {rce3_motion:.4f} | {rce3_zoom:.4f} | {rce3_elastic:.4f} | {rce3_quant:.4f} | {rce3_gaussian:.4f} | {rce3_impulse:.4f} | {rce3_shot:.4f} | {rce3_iso:.4f} | {rce3_pixelate:.4f} | {rce3_jpeg:.4f} |
| $\\text{{RR}}_{{3}}$  |      -       | {rr3_bright:.4f}  | {rr3_dark:.4f}  | {rr3_fog:.4f}  | {rr3_frost:.4f}  | {rr3_snow:.4f}  | {rr3_contrast:.4f}  | {rr3_defocus:.4f}  | {rr3_glass:.4f}  | {rr3_motion:.4f}  | {rr3_zoom:.4f}  | {rr3_elastic:.4f}  | {rr3_quant:.4f}  | {rr3_gaussian:.4f}  | {rr3_impulse:.4f}  | {rr3_shot:.4f}  | {rr3_iso:.4f}  | {rr3_pixelate:.4f}  | {rr3_jpeg:.4f}  |

- **Summary:** $\\text{{mCE}}_1 =$ {mce1:.4f}, $\\text{{RmCE}}_1 =$ {rmce1:.4f}, $\\text{{mRR}}_1 =$ {mrr1:.4f}, $\\text{{mCE}}_2 =$ {mce2:.4f}, $\\text{{RmCE}}_2 =$ {rmce2:.4f}, $\\text{{mRR}}_2 =$ {mrr2:.4f}, $\\text{{mCE}}_3 =$ {mce3:.4f}, $\\text{{RmCE}}_3 =$ {rmce3:.4f}, $\\text{{mRR}}_3 =$ {mrr3:.4f}

""".format(
    # DEE1 = abs_rel - a1 + 1
    clean1 = clean1,
    bright1 = bright1, dark1 = dark1, fog1 = fog1, frost1 = frost1, snow1 = snow1, contrast1 = contrast1,
    defocus1 = defocus1, glass1 = glass1, motion1 = motion1, zoom1 = zoom1, elastic1 = elastic1, quant1 = quant1,
    gaussian1 = gaussian1, impulse1 = impulse1, shot1 = shot1, iso1 = iso1, pixelate1 = pixelate1, jpeg1 = jpeg1,
    # DEE2 = 0.5 * (abs_rel - a1 + 1)
    clean2 = clean2,
    bright2 = bright2, dark2 = dark2, fog2 = fog2, frost2 = frost2, snow2 = snow2, contrast2 = contrast2,
    defocus2 = defocus2, glass2 = glass2, motion2 = motion2, zoom2 = zoom2, elastic2 = elastic2, quant2 = quant2,
    gaussian2 = gaussian2, impulse2 = impulse2, shot2 = shot2, iso2 = iso2, pixelate2 = pixelate2, jpeg2 = jpeg2,
    # DEE3 = abs_rel / a1
    clean3 = clean3,
    bright3 = bright3, dark3 = dark3, fog3 = fog3, frost3 = frost3, snow3 = snow3, contrast3 = contrast3,
    defocus3 = defocus3, glass3 = glass3, motion3 = motion3, zoom3 = zoom3, elastic3 = elastic3, quant3 = quant3,
    gaussian3 = gaussian3, impulse3 = impulse3, shot3 = shot3, iso3 = iso3, pixelate3 = pixelate3, jpeg3 = jpeg3,
    # CE (DEE1)
    ce1_bright = ce1_bright, ce1_dark = ce1_dark, ce1_fog = ce1_fog, ce1_frost = ce1_frost, ce1_snow = ce1_snow, ce1_contrast = ce1_contrast,
    ce1_defocus = ce1_defocus, ce1_glass = ce1_glass, ce1_motion = ce1_motion, ce1_zoom = ce1_zoom, ce1_elastic = ce1_elastic, ce1_quant = ce1_quant,
    ce1_gaussian = ce1_gaussian, ce1_impulse = ce1_impulse, ce1_shot = ce1_shot, ce1_iso = ce1_iso, ce1_pixelate = ce1_pixelate, ce1_jpeg = ce1_jpeg,
    # RCE (DEE1)
    rce1_bright = rce1_bright, rce1_dark = rce1_dark, rce1_fog = rce1_fog, rce1_frost = rce1_frost, rce1_snow = rce1_snow, rce1_contrast = rce1_contrast,
    rce1_defocus = rce1_defocus, rce1_glass = rce1_glass, rce1_motion = rce1_motion, rce1_zoom = rce1_zoom, rce1_elastic = rce1_elastic, rce1_quant = rce1_quant,
    rce1_gaussian = rce1_gaussian, rce1_impulse = rce1_impulse, rce1_shot = rce1_shot, rce1_iso = rce1_iso, rce1_pixelate = rce1_pixelate, rce1_jpeg = rce1_jpeg,
    # RR (DEE1)
    rr1_bright = rr1_bright, rr1_dark = rr1_dark, rr1_fog = rr1_fog, rr1_frost = rr1_frost, rr1_snow = rr1_snow, rr1_contrast = rr1_contrast,
    rr1_defocus = rr1_defocus, rr1_glass = rr1_glass, rr1_motion = rr1_motion, rr1_zoom = rr1_zoom, rr1_elastic = rr1_elastic, rr1_quant = rr1_quant,
    rr1_gaussian = rr1_gaussian, rr1_impulse = rr1_impulse, rr1_shot = rr1_shot, rr1_iso = rr1_iso, rr1_pixelate = rr1_pixelate, rr1_jpeg = rr1_jpeg,
    # CE (DEE2)
    ce2_bright = ce2_bright, ce2_dark = ce2_dark, ce2_fog = ce2_fog, ce2_frost = ce2_frost, ce2_snow = ce2_snow, ce2_contrast = ce2_contrast,
    ce2_defocus = ce2_defocus, ce2_glass = ce2_glass, ce2_motion = ce2_motion, ce2_zoom = ce2_zoom, ce2_elastic = ce2_elastic, ce2_quant = ce2_quant,
    ce2_gaussian = ce2_gaussian, ce2_impulse = ce2_impulse, ce2_shot = ce2_shot, ce2_iso = ce2_iso, ce2_pixelate = ce2_pixelate, ce2_jpeg = ce2_jpeg,
    # RCE (DEE2)
    rce2_bright = rce2_bright, rce2_dark = rce2_dark, rce2_fog = rce2_fog, rce2_frost = rce2_frost, rce2_snow = rce2_snow, rce2_contrast = rce2_contrast,
    rce2_defocus = rce2_defocus, rce2_glass = rce2_glass, rce2_motion = rce2_motion, rce2_zoom = rce2_zoom, rce2_elastic = rce2_elastic, rce2_quant = rce2_quant,
    rce2_gaussian = rce2_gaussian, rce2_impulse = rce2_impulse, rce2_shot = rce2_shot, rce2_iso = rce2_iso, rce2_pixelate = rce2_pixelate, rce2_jpeg = rce2_jpeg,
    # RR (DEE2)
    rr2_bright = rr2_bright, rr2_dark = rr2_dark, rr2_fog = rr2_fog, rr2_frost = rr2_frost, rr2_snow = rr2_snow, rr2_contrast = rr2_contrast,
    rr2_defocus = rr2_defocus, rr2_glass = rr2_glass, rr2_motion = rr2_motion, rr2_zoom = rr2_zoom, rr2_elastic = rr2_elastic, rr2_quant = rr2_quant,
    rr2_gaussian = rr2_gaussian, rr2_impulse = rr2_impulse, rr2_shot = rr2_shot, rr2_iso = rr2_iso, rr2_pixelate = rr2_pixelate, rr2_jpeg = rr2_jpeg,
    # CE (DEE3)
    ce3_bright = ce3_bright, ce3_dark = ce3_dark, ce3_fog = ce3_fog, ce3_frost = ce3_frost, ce3_snow = ce3_snow, ce3_contrast = ce3_contrast,
    ce3_defocus = ce3_defocus, ce3_glass = ce3_glass, ce3_motion = ce3_motion, ce3_zoom = ce3_zoom, ce3_elastic = ce3_elastic, ce3_quant = ce3_quant,
    ce3_gaussian = ce3_gaussian, ce3_impulse = ce3_impulse, ce3_shot = ce3_shot, ce3_iso = ce3_iso, ce3_pixelate = ce3_pixelate, ce3_jpeg = ce3_jpeg,
    # RCE (DEE3)
    rce3_bright = rce3_bright, rce3_dark = rce3_dark, rce3_fog = rce3_fog, rce3_frost = rce3_frost, rce3_snow = rce3_snow, rce3_contrast = rce3_contrast,
    rce3_defocus = rce3_defocus, rce3_glass = rce3_glass, rce3_motion = rce3_motion, rce3_zoom = rce3_zoom, rce3_elastic = rce3_elastic, rce3_quant = rce3_quant,
    rce3_gaussian = rce3_gaussian, rce3_impulse = rce3_impulse, rce3_shot = rce3_shot, rce3_iso = rce3_iso, rce3_pixelate = rce3_pixelate, rce3_jpeg = rce3_jpeg,
    # RR (DEE3)
    rr3_bright = rr3_bright, rr3_dark = rr3_dark, rr3_fog = rr3_fog, rr3_frost = rr3_frost, rr3_snow = rr3_snow, rr3_contrast = rr3_contrast,
    rr3_defocus = rr3_defocus, rr3_glass = rr3_glass, rr3_motion = rr3_motion, rr3_zoom = rr3_zoom, rr3_elastic = rr3_elastic, rr3_quant = rr3_quant,
    rr3_gaussian = rr3_gaussian, rr3_impulse = rr3_impulse, rr3_shot = rr3_shot, rr3_iso = rr3_iso, rr3_pixelate = rr3_pixelate, rr3_jpeg = rr3_jpeg,
    # mCE & RmCE & mRR (DEE1)
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
    mrr1 = np.mean(
        [rr1_bright, rr1_dark, rr1_fog, rr1_frost, rr1_snow, rr1_contrast,
        rr1_defocus, rr1_glass, rr1_motion, rr1_zoom, rr1_elastic, rr1_quant,
        rr1_gaussian, rr1_impulse, rr1_shot, rr1_iso, rr1_pixelate, rr1_jpeg,]
    ),
    # mCE & RmCE & mRR (DEE2)
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
    mrr2 = np.mean(
        [rr2_bright, rr2_dark, rr2_fog, rr2_frost, rr2_snow, rr2_contrast,
        rr2_defocus, rr2_glass, rr2_motion, rr2_zoom, rr2_elastic, rr2_quant,
        rr2_gaussian, rr2_impulse, rr2_shot, rr2_iso, rr2_pixelate, rr2_jpeg,]
    ),
    # mCE & RmCE & mRR (DEE3)
    mce3 = np.mean(
        [ce3_bright, ce3_dark, ce3_fog, ce3_frost, ce3_snow, ce3_contrast,
        ce3_defocus, ce3_glass, ce3_motion, ce3_zoom, ce3_elastic, ce3_quant,
        ce3_gaussian, ce3_impulse, ce3_shot, ce3_iso, ce3_pixelate, ce3_jpeg,]
    ),
    rmce3 = np.mean(
        [rce3_bright, rce3_dark, rce3_fog, rce3_frost, rce3_snow, rce3_contrast,
        rce3_defocus, rce3_glass, rce3_motion, rce3_zoom, rce3_elastic, rce3_quant,
        rce3_gaussian, rce3_impulse, rce3_shot, rce3_iso, rce3_pixelate, rce3_jpeg,]
    ),
    mrr3 = np.mean(
        [rr3_bright, rr3_dark, rr3_fog, rr3_frost, rr3_snow, rr3_contrast,
        rr3_defocus, rr3_glass, rr3_motion, rr3_zoom, rr3_elastic, rr3_quant,
        rr3_gaussian, rr3_impulse, rr3_shot, rr3_iso, rr3_pixelate, rr3_jpeg,]
    ),
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
