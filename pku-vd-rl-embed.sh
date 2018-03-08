#!/bin/sh

export CUDA_VISIBLE_DEVICES=4
source ../triplet-reid-rl-attention/venv/bin/activate

epoch=4000
dataset_size=''
# expr_dir='expr_attention_euclidean_fc1024_inception_finetune_rl_sample_para_3'
expr_dir='expr_attention_euclidean_fc1024_inception_finetune_rl_sample_para_2'
# expr_dir='expr_attention_euclidean_fc1024_inception_finetune_rl_sample_para_1'
# expr_dir='expr_attention_euclidean_fc1024_inception_finetune_rl_sample_para'
# expr_dir='expr_attention_euclidean_fc1024_inception_finetune_rl_heavy_agent'

python embed_rl.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128 \
        --rl_activation sigmoid

dataset_size='small_'

python embed_rl.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128 \
        --rl_activation sigmoid

python ./evaluate.py \
    --excluder diagonal \
    --query_dataset ./data/pku-vd/VD1_query.csv \
    --query_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_query_${epoch}_embeddings.h5 \
    --gallery_dataset ./data/pku-vd/VD1_${dataset_size}query.csv \
    --gallery_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
    --metric euclidean \
    --filename ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_evaluation.json \
    --batch_size 64 \
    
