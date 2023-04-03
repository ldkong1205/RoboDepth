CUDA_VISIBLE_DEVICES=0 python3 zoo/RA-Depth/evaluate_depth_kittic.py \
    --load_weights_folder models/RA-Depth \
    --eval_corr_type 'all' \
    --eval_mono \
    --height 192 \
    --width 640 \
    --scales 0 \
    --data_path kitti_data \
    --png