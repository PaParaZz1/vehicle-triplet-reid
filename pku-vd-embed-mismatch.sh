#!/bin/sh

export CUDA_VISIBLE_DEVICES=7
source venv/bin/activate

dataset_size=''
# epoch=450000
# expr_dir='expr_attention_euclidean_fc1024_inception_MBA_5b_addition_inception_1.0_1_finetune_0'

epoch=285886
expr_dir='ckpt_inception_v4'
DATASET='VD1'

python embed.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/${DATASET}_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128

dataset_size='small_'

python embed.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/${DATASET}_${dataset_size}query.csv \
        --filename pku-vd_${DATASET}_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128

python evaluate_mismatch.py \
    --excluder diagonal \
    --query_dataset ./data/pku-vd/${DATASET}_query.csv \
    --query_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_${DATASET}_query_${epoch}_embeddings.h5 \
    --gallery_dataset ./data/pku-vd/${DATASET}_${dataset_size}query.csv \
    --gallery_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_${DATASET}_${dataset_size}query_${epoch}_embeddings.h5 \
    --metric euclidean \
    --filename ./experiments/pku-vd/${expr_dir}/pku-vd_${DATASET}_${dataset_size}query_${epoch}_evaluation.json \
    --batch_size 64 \
    
