#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

TYPE="query"

# --experiment_root ./experiments/pku-vd/expr_attention_euclidean_fc1024_inception_mixed_attention_inception_finetune_0 \
python embed.py \
        --experiment_root ./experiments/pku-vd/expr_attention_euclidean_fc1024_inception_multi-residual-head_attention_5_branch_inception_10.0_finetune_0 \
        --dataset data/VeRi/VeRi_${TYPE}.csv \
        --filename VeRi_${TYPE}_370000_embeddings.h5 \
        --checkpoint checkpoint-370000 \
        --batch_size 128
        # --flip_augment \
        # --crop_augment five \
        # --aggregator mean
