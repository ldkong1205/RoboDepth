CUDA_VISIBLE_DEVICES=0 python3 zoo/MonoDepth2/evaluate_kittic.py \
    --num_layers 18 \
    --load_weights_folder models/MonoDepth2/stereo_640x192 \
    --eval_stereo \
    --eval_corr_type 'all' \
    --num_workers 8 \