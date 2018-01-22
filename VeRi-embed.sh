#!/bin/sh

python embed.py \
        --experiment_root ./experiments/VeRi/expr1 \
        --dataset data/VeRi/VeRi_test.csv \
        --filename VeRi_test_50000_embeddings.h5 \
        --checkpoint checkpoint-50000 \
        --batch_size 16
        # --flip_augment \
        # --crop_augment five \
        # --aggregator mean
