CUDA_VISIBLE_DEVICES=0 python3 zoo/MonoViT/evaluate_depth_kittic.py \
    --data_path kitti_data \
    --eval_corr_type 'all' \
    --load_weights_folder models/MonoViT/MonoViT_MS_640x192 \
    --eval_mono
    