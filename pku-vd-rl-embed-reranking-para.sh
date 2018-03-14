#!/bin/sh

export CUDA_VISIBLE_DEVICES=6
source ../triplet-reid-rl-attention/venv/bin/activate

epoch=6000
dataset_size='small_'
HIDDEN_UNITS=256
expr_dir='expr_attention_euclidean_fc1024_inception_mixed_attention_inception_rl_256_ft_1'
# expr_dir='expr_attention_euclidean_fc1024_inception_mixed_attention_inception_rl_128_ft_0'
# expr_dir='expr_attention_euclidean_fc1024_inception_mixed_attention_inception_rl_512_ft_0'
# expr_dir='expr_attention_euclidean_fc1024_inception_mixed_attention_inception_finetune_rl_256_2'

function embs_sup () {
    export CUDA_VISIBLE_DEVICES=$1
    source ../triplet-reid-rl-attention/venv/bin/activate
    dataset_size=''
    epoch=$2
    HIDDEN_UNITS=$3
    expr_dir=$4

    python embed_rl_reranking.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128 \
        --rl_activation sigmoid \
        --rl_hidden_units ${HIDDEN_UNITS} \
        --emb_type sup 

    dataset_size=$5

    python embed_rl_reranking.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128 \
        --rl_activation sigmoid \
        --rl_hidden_units ${HIDDEN_UNITS} \
        --emb_type sup
}

function embs_rl () {
    export CUDA_VISIBLE_DEVICES=$1
    source ../triplet-reid-rl-attention/venv/bin/activate
    dataset_size=''
    epoch=$2
    HIDDEN_UNITS=$3
    expr_dir=$4

    python embed_rl_reranking.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128 \
        --rl_activation sigmoid \
        --rl_hidden_units ${HIDDEN_UNITS} \
        --emb_type rl

    dataset_size=$5

    python embed_rl_reranking.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128 \
        --rl_activation sigmoid \
        --rl_hidden_units ${HIDDEN_UNITS} \
        --emb_type rl
}

embs_sup 3 6000 256 'expr_attention_euclidean_fc1024_inception_mixed_attention_inception_rl_256_ft_1' ${dataset_size} &
embs_rl 4 6000 256 'expr_attention_euclidean_fc1024_inception_mixed_attention_inception_rl_256_ft_1' ${dataset_size} &

wait

python evaluate_reranking.py \
    --excluder diagonal \
    --query_dataset ./data/pku-vd/VD1_query.csv \
    --query_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_query_${epoch}_embeddings.h5 \
    --gallery_dataset ./data/pku-vd/VD1_${dataset_size}query.csv \
    --gallery_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
    --metric euclidean \
    --filename ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_evaluation.json \
    --batch_size 8 \
    
