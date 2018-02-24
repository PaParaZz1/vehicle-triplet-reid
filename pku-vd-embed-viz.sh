#!/bin/sh

export CUDA_VISIBLE_DEVICES=2
epoch=84000
dataset_size=''
expr_dir='expr_attention_euclidean_fc1024_spatial_attention_softmax_resnet_v2_50_finetune'
# expr_dir='expr_attention_euclidean_fc1024_spatial_attention_resnet_v2_50_0'

python embed_viz.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings_viz.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128

