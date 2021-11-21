#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0
python validate.py \
-gen_bs 128 \
-dis_bs 64 \
--dataset cifar10 \
--img_size 32 \
--max_iter 100000 \
--model sngan_cifar10 \
--latent_dim 128 \
--gf_dim 256 \
--df_dim 128 \
--g_spectral_norm False \
--d_spectral_norm True \
--g_lr 0.0002 \
--d_lr 0.0002 \
--beta1 0.0 \
--beta2 0.9 \
--init_type xavier_uniform \
--n_critic 8 \
--val_freq 20 \
--exp_name multi-sngan \
--load_path logs/multiD_cifar10_2020_09_16_07_52_58/