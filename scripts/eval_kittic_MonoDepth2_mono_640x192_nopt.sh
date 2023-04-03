CUDA_VISIBLE_DEVICES=0 python3 zoo/MonoDepth2/evaluate_kittic.py \
    --num_layers 18 \
    --load_weights_folder models/MonoDepth2/mono_no_pt_640x192 \
    --eval_mono \
    --eval_corr_type 'all' \
    --num_workers 8 \