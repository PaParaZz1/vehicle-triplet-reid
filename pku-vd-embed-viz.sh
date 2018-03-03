#!/bin/sh

export CUDA_VISIBLE_DEVICES=6
epoch=150000
dataset_size=''
expr_dir='expr_attention_euclidean_fc1024_inception_multi-residual-head_attention_inception_finetune_0'

python embed_viz.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings_viz.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128

