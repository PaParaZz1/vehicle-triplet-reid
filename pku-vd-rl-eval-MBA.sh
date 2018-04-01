#!/bin/sh

export CUDA_VISIBLE_DEVICES=5
source ../triplet-reid-rl-attention/venv/bin/activate

epoch=10000
dataset_size=''
# HIDDEN_UNITS="128 64 128"
# expr_dir='expr_attention_euclidean_fc1024_inception_MBA_5b_addition_inception_rl_MLP_ft_4'
HIDDEN_UNITS="256 128 256"
expr_dir='expr_attention_euclidean_fc1024_inception_MBA_5b_addition_inception_rl_MLP_ft_6'

dataset_size='small_'

python ./evaluate_rl_MBA.py \
    --experiment_root ./experiments/pku-vd/${expr_dir} \
    --checkpoint checkpoint-${epoch} \
    --excluder diagonal \
    --query_dataset ./data/pku-vd/VD1_query.csv \
    --query_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_query_${epoch}_embeddings.h5 \
    --gallery_dataset ./data/pku-vd/VD1_${dataset_size}query.csv \
    --gallery_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
    --metric euclidean \
    --rl_hidden_units ${HIDDEN_UNITS} \
    --rl_activation sigmoid \
    --filename ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_evaluation.json \
    --batch_size 32 \
    
