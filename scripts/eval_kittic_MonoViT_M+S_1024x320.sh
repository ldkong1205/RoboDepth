CUDA_VISIBLE_DEVICES=0 python3 zoo/MonoViT/evaluate_hr_depth_kittic.py \
    --data_path kitti_data \
    --eval_corr_type 'all' \
    --load_weights_folder models/MonoViT/MonoViT_MS_1024x320 \
    --height 320 \
    --width 1024 \
    --eval_mono
    