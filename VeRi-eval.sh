#!/bin/sh

export CUDA_VISIBLE_DEVICES=0

ITER=370000
EXPR_ROOT='./experiments/pku-vd/expr_attention_euclidean_fc1024_inception_multi-residual-head_attention_5_branch_inception_10.0_finetune_0'

python ./evaluate.py \
    --excluder diagonal\
    --query_dataset ./data/VeRi/VeRi_query.csv \
    --query_embeddings ${EXPR_ROOT}/VeRi_query_${ITER}_embeddings.h5 \
    --gallery_dataset ./data/VeRi/VeRi_test.csv \
    --gallery_embeddings ${EXPR_ROOT}/VeRi_test_${ITER}_embeddings.h5 \
    --metric euclidean\
    --filename ${EXPR_ROOT}/VeRi_${ITER}_evaluation.json \
    --batch_size 64\
    
