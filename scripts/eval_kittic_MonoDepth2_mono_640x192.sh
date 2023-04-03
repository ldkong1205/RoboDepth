CUDA_VISIBLE_DEVICES=0 python3 zoo/MonoDepth2/evaluate_kittic.py \
    --num_layers 50 \
    --load_weights_folder models/monodepth2/logs/0221/mono_res50/mono_model_res50/models/weights_14 \
    --eval_mono \
    --eval_corr_type 'all' \
    --num_workers 8 \