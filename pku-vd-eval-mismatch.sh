#!/bin/sh

export CUDA_VISIBLE_DEVICES=7
source venv/bin/activate

expr_dir='ckpt_inception_v4'
epoch=285886
DATASET='VD1'

dataset_size='small_'

python evaluate_mismatch.py \
    --excluder diagonal \
    --query_dataset ./data/pku-vd/${DATASET}_query.csv \
    --query_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_${DATASET}_query_${epoch}_embeddings.h5 \
    --gallery_dataset ./data/pku-vd/${DATASET}_${dataset_size}query.csv \
    --gallery_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_${DATASET}_${dataset_size}query_${epoch}_embeddings.h5 \
    --metric euclidean \
    --filename ./experiments/pku-vd/${expr_dir}/pku-vd_${DATASET}_${dataset_size}query_${epoch}_evaluation.json \
    --batch_size 64 \
    
