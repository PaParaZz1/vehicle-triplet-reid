#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
source ./venv/bin/activate

epoch=450000
expr_dir='expr_attention_euclidean_fc1024_inception_MBA_5b_addition_inception_1.0_1_finetune_0'
# dataset_size='large_'
# dataset_size='medium_'
dataset_size='small_'

# python ./evaluate.py \
python ./evaluate_timing.py \
    --excluder diagonal \
    --query_dataset ./data/pku-vd/VD1_query.csv \
    --query_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_query_${epoch}_embeddings.h5 \
    --gallery_dataset ./data/pku-vd/VD1_${dataset_size}query.csv \
    --gallery_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
    --metric euclidean \
    --filename ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_evaluation.json \
    --batch_size 64
