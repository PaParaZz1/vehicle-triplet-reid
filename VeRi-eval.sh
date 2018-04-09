#!/bin/sh

export CUDA_VISIBLE_DEVICES=6
source venv/bin/activate
expr_dir='expr_attention_euclidean_fc1024_inception_MBA_5b_addition_inception_6'
epoch=220000
# expr_dir='expr_attention_euclidean_fc1024_inception_scratch'
# epoch=120000

DATASET='_track'

python ./evaluate.py \
    --excluder veri_track\
    --query_dataset ./data/VeRi${DATASET}/VeRi_query.csv \
    --query_embeddings ./experiments/VeRi/${expr_dir}/VeRi_query_${epoch}_embeddings.h5 \
    --gallery_dataset ./data/VeRi${DATASET}/VeRi_test.csv \
    --gallery_embeddings ./experiments/VeRi/${expr_dir}/VeRi_test_${epoch}_embeddings.h5 \
    --metric euclidean\
    --filename ./experiments/VeRi/${expr_dir}/VeRi_${epoch}_evaluation.json \
    --batch_size 64\
