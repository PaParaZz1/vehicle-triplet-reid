#!/bin/sh

# epoch=360000
epoch=260000
# dataset_size=''
dataset_size='small_'
# expr_dir='expr_resnet50_v2_50_1'
expr_dir='expr_inception_v4'

python embed.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 64
        # --flip_augment \
        # --crop_augment five \
        # --aggregator mean
