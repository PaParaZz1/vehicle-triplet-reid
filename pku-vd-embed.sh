#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
epoch=20000
# dataset_size=''
dataset_size='small_'
# expr_dir='expr_cls_euclidean_1e-2_balanced_resnet-50'
# expr_dir='expr_cls_euclidean_1.0_fc1024_cls_wo_projection_resnet_v2_50'
# expr_dir='expr_cls_euclidean_0.5_fc1024_cls_projection_resnet_v2_50__2'
# expr_dir='expr_cls_euclidean_0.5_fc1024_cls_projection_resnet_v2_50'
expr_dir='expr_cls_euclidean_1.0_fc1024_cls_projection_resnet_v2_50_0'

python embed.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128
        # --flip_augment \
        # --crop_augment five \
        # --aggregator mean
