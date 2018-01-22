#!/bin/sh

epoch=50000
dataset_size='small_'

python ./evaluate.py \
    --excluder diagonal \
    --query_dataset ./data/pku-vd/VD1_query.csv \
    --query_embeddings ./experiments/pku-vd/expr_resnet50/pku-vd_VD1_query_${epoch}_embeddings.h5 \
    --gallery_dataset ./data/pku-vd/VD1_${dataset_size}query.csv \
    --gallery_embeddings ./experiments/pku-vd/expr_resnet50/pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
    --metric euclidean \
    --filename ./experiments/pku-vd/expr_resnet50/pku-vd_VD1_${dataset_size}${epoch}_evaluation.json \
    --batch_size 32 \
    # --batch_size 128 \
    
