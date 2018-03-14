#!/bin/sh

export CUDA_VISIBLE_DEVICES=7
source ../triplet-reid-rl-attention/venv/bin/activate

epoch=1000
dataset_size=''
# HIDDEN_UNITS="256 128"
# expr_dir='expr_attention_euclidean_fc1024_inception_mixed_attention_inception_rl_MLP_ft_3'
# HIDDEN_UNITS="128"
# expr_dir='expr_attention_euclidean_fc1024_inception_mixed_attention_inception_rl_128_ft_2'
HIDDEN_UNITS="128 64 128"
expr_dir='expr_attention_euclidean_fc1024_inception_mixed_attention_inception_rl_MLP_ft_4'


python embed_rl_reranking.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128 \
        --rl_activation sigmoid \
        --rl_hidden_units ${HIDDEN_UNITS} \
        --emb_type sup 

python embed_rl_reranking.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128 \
        --rl_activation sigmoid \
        --rl_hidden_units ${HIDDEN_UNITS} \
        --emb_type rl

dataset_size='small_'
# dataset_size='medium_'
# dataset_size='large_'

python embed_rl_reranking.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128 \
        --rl_activation sigmoid \
        --rl_hidden_units ${HIDDEN_UNITS} \
        --emb_type sup

python embed_rl_reranking.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128 \
        --rl_activation sigmoid \
        --rl_hidden_units ${HIDDEN_UNITS} \
        --emb_type rl

python evaluate_reranking.py \
    --excluder diagonal \
    --query_dataset ./data/pku-vd/VD1_query.csv \
    --query_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_query_${epoch}_embeddings.h5 \
    --gallery_dataset ./data/pku-vd/VD1_${dataset_size}query.csv \
    --gallery_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
    --metric euclidean \
    --filename ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_evaluation.json \
    --batch_size 8 \
    --reranking_type combine \
    --reranking_topk 50
    
