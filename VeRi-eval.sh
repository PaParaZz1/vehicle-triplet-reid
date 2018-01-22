#!/bin/sh

python ./evaluate.py \
    --excluder diagonal\
    --query_dataset ./data/VeRi/VeRi_query.csv \
    --query_embeddings ./experiments/VeRi/expr1/VeRi_query_50000_embeddings.h5 \
    --gallery_dataset ./data/VeRi/VeRi_test.csv \
    --gallery_embeddings ./experiments/VeRi/expr1/VeRi_test_50000_embeddings.h5 \
    --metric euclidean\
    --filename ./experiments/VeRi/expr1/VeRi_50000_evaluation.json \
    --batch_size 16\
    
