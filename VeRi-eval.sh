#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
source venv/bin/activate
expr_dir='expr_attention_euclidean_fc1024_inception_scratch'
epoch=20000

python ./evaluate.py \
    --excluder veri\
    --query_dataset ./data/VeRi/VeRi_query.csv \
    --query_embeddings ./experiments/VeRi/${expr_dir}/VeRi_query_${epoch}_embeddings.h5 \
    --gallery_dataset ./data/VeRi/VeRi_test.csv \
    --gallery_embeddings ./experiments/VeRi/${expr_dir}/VeRi_test_${epoch}_embeddings.h5 \
    --metric euclidean\
    --filename ./experiments/VeRi/${expr_dir}/VeRi_${epoch}_evaluation.json \
    --batch_size 64\
    
