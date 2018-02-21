#!/bin/sh

export CUDA_VISIBLE_DEVICES=2
epoch=320000
# dataset_size=''
dataset_size='small_'
expr_dir='expr_attention_euclidean_fc1024_spatial_attention_resnet_v2_50_0'
# expr_dir='expr_cls_euclidean_1.0_fc1024_cls_resnet_v2_50_1'
# expr_dir='expr_cls_euclidean_1.0_fc1024_cls_projection_resnet_v2_50_anchorwise_cls_0'

python embed.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128
        # --flip_augment \
        # --crop_augment five \
        # --aggregator mean
