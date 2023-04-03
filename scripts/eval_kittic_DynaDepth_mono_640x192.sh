CUDA_VISIBLE_DEVICES=0 python3 zoo/DynaDepth/evaluate_depth_kittic.py \
    --data_path kitti_data \
    --num_layers 50 \
    --load_weights_folder models/DynaDepth/R50 \
    --eval_mono