#!/bin/sh

export CUDA_VISIBLE_DEVICES=6
source venv/bin/activate

expr_dir='expr_attention_euclidean_fc1024_vgg_MBA_5b_kl_addition_vgg_16_0'
epoch=12000

TYPE="query"
DATASET='_track'

python embed_timing.py \
        --experiment_root ./experiments/VeRi/${expr_dir} \
        --dataset data/VeRi${DATASET}/VeRi_${TYPE}.csv \
        --filename VeRi_${TYPE}_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 1
        --flip_augment \
        --aggregator mean \
#         --crop_augment five \

TYPE='test'

python embed_timing.py \
        --experiment_root ./experiments/VeRi/${expr_dir} \
        --dataset data/VeRi${DATASET}/VeRi_${TYPE}.csv \
        --filename VeRi_${TYPE}_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --flip_augment \
        --aggregator mean \
        --batch_size 1
#        --crop_augment five \

python ./evaluate.py \
    --excluder veri_track\
    --query_dataset ./data/VeRi${DATASET}/VeRi_query.csv \
    --query_embeddings ./experiments/VeRi/${expr_dir}/VeRi_query_${epoch}_embeddings.h5 \
    --gallery_dataset ./data/VeRi${DATASET}/VeRi_test.csv \
    --gallery_embeddings ./experiments/VeRi/${expr_dir}/VeRi_test_${epoch}_embeddings.h5 \
    --metric euclidean\
    --filename ./experiments/VeRi/${expr_dir}/VeRi_${epoch}_evaluation.json \
    --batch_size 64\
    # --excluder diagonal\
