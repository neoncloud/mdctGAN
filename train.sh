#!/bin/bash

python train.py \
    --name your_training_name \
    --dataroot /home/neoncloud/VCTK-Corpus/train.csv --evalroot /home/neoncloud/VCTK-Corpus/test.csv \
    --lr_sampling_rate 16000 --sr_sampling_rate 48000 \
    --batchSize 20 \
    --gpu_id 0 --fp16 --nThreads 16 --lr 1.5e-4 \
    --arcsinh_transform --abs_spectro --arcsinh_gain 1000 --center \
    --norm_range -1 1 --smooth 0.0 --abs_norm --src_range -5 5 \
    --netG local --ngf 56 \
    --n_downsample_global 3 --n_blocks_global 4 \
    --n_blocks_attn_g 3 --dim_head_g 128 --heads_g 6 --proj_factor_g 4 \
    --n_blocks_attn_l 0 --n_blocks_local 3 \
    --fit_residual --upsample_type interpolate --downsample_type resconv \
    --niter 60 --niter_decay 60 --num_D 3 \
    --eval_freq 32000 --save_latest_freq 16000 --save_epoch_freq 10 --display_freq 16000 --tf_log