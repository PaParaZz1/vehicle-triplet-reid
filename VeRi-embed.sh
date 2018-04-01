#!/bin/sh

export CUDA_VISIBLE_DEVICES=4
source venv/bin/activate
expr_dir='expr_attention_euclidean_fc1024_inception_scratch'
epoch=140000

TYPE="query"

python embed.py \
        --experiment_root ./experiments/VeRi/${expr_dir} \
        --dataset data/VeRi/VeRi_${TYPE}.csv \
        --filename VeRi_${TYPE}_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128

TYPE='test'

python embed.py \
        --experiment_root ./experiments/VeRi/${expr_dir} \
        --dataset data/VeRi/VeRi_${TYPE}.csv \
        --filename VeRi_${TYPE}_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128

python ./evaluate.py \
    --excluder veri\
    --query_dataset ./data/VeRi/VeRi_query.csv \
    --query_embeddings ./experiments/VeRi/${expr_dir}/VeRi_query_${epoch}_embeddings.h5 \
    --gallery_dataset ./data/VeRi/VeRi_test.csv \
    --gallery_embeddings ./experiments/VeRi/${expr_dir}/VeRi_test_${epoch}_embeddings.h5 \
    --metric euclidean\
    --filename ./experiments/VeRi/${expr_dir}/VeRi_${epoch}_evaluation.json \
    --batch_size 64\
    
