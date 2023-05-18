#!/bin/bash

python generate_audio.py \
    --name output_folder_name \
    --load_pretrain /home/neoncloud/pix2pixHDAudioSR/checkpoints/vctk_fintune_G4A3L3_56ngf_3x \
    --lr_sampling_rate 16000 --sr_sampling_rate 48000 \
    --dataroot /mnt/d/Desktop/LJ001-0056.wav --batchSize 16 \
    --gpu_id 0 --fp16 --nThreads 1 \
    --arcsinh_transform --abs_spectro --arcsinh_gain 1000 --center \
    --norm_range -1 1 --smooth 0.0 --abs_norm --src_range -5 5 \
    --netG local --ngf 56 --niter 40 \
    --n_downsample_global 3 --n_blocks_global 4 \
    --n_blocks_attn_g 3 --dim_head_g 128 --heads_g 6 --proj_factor_g 4 \
    --n_blocks_attn_l 0 --n_blocks_local 3 --gen_overlap 0 \
    --fit_residual --upsample_type interpolate --downsample_type resconv --phase test