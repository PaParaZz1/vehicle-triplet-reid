#!/bin/sh

export CUDA_VISIBLE_DEVICES=4

python embed.py \
        --experiment_root ./experiments/pku-vd/expr_attention_euclidean_fc1024_inception_mixed_attention_inception_finetune_0 \
        --dataset data/VeRi/VeRi_test.csv \
        --filename VeRi_test_50000_embeddings.h5 \
        --checkpoint checkpoint-50000 \
        --batch_size 16
        # --flip_augment \
        # --crop_augment five \
        # --aggregator mean
