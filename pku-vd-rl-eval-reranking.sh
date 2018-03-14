#!/bin/sh

export CUDA_VISIBLE_DEVICES=3
TOPK=100
source ../triplet-reid-rl-attention/venv/bin/activate

epoch=1000
dataset_size='small_'
# dataset_size='medium_'
# dataset_size='large_'

# HIDDEN_UNITS="128 64"
# expr_dir='expr_attention_euclidean_fc1024_inception_mixed_attention_inception_rl_MLP_ft_2'
HIDDEN_UNITS="128 64 128"
expr_dir='expr_attention_euclidean_fc1024_inception_mixed_attention_inception_rl_MLP_ft_4'

python evaluate_reranking.py \
    --excluder diagonal \
    --query_dataset ./data/pku-vd/VD1_query.csv \
    --query_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_query_${epoch}_embeddings.h5 \
    --gallery_dataset ./data/pku-vd/VD1_${dataset_size}query.csv \
    --gallery_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
    --metric euclidean \
    --filename ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_evaluation.json \
    --batch_size 16 \
    --reranking_type topk \
    --reranking_topk ${TOPK}
    
