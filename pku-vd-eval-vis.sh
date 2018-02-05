#!/bin/sh

epoch=360000
dataset_size='small_'
expr_dir='pku-vd_resnet50_v2_results'

python ./evaluate.py \
    --excluder diagonal \
    --query_dataset ./data/pku-vd/VD1_query.csv \
    --query_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_query_${epoch}_embeddings.h5 \
    --gallery_dataset ./data/pku-vd/VD1_${dataset_size}query.csv \
    --gallery_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
    --filename ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_evaluation.json \
    --batch_size 128 \
    --metric cosine \
    # --metric zscore_norm \
#     --display
