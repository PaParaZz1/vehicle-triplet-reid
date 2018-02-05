#!/bin/sh

epoch=250000
# dataset_size=''
dataset_size='small_'
expr_dir='expr_cls_euclidean_1.0_resnet-50'
# expr_dir='expr_cls_euclidean_1e-3_balanced_resnet-50'

python embed.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_augment_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128 \
        --flip_augment \
        --crop_augment five \
        --aggregator mean \
