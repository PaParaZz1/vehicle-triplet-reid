#!/bin/sh

epoch=50000
dataset_size=''

python embed.py \
        --experiment_root ./experiments/pku-vd/expr_resnet50_v2_50_1 \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 32
        # --flip_augment \
        # --crop_augment five \
        # --aggregator mean
