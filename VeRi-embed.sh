#!/bin/sh

export CUDA_VISIBLE_DEVICES=7
source venv/bin/activate
# expr_dir='expr_attention_euclidean_fc1024_inception_MBA_5b_addition_inception_6'
# epoch=240000
# expr_dir='expr_attention_euclidean_fc1024_inception_scratch'
# epoch=120000
expr_dir='expr_attention_euclidean_fc1024_MBA_5b_addition_resnet_v2_50_10.0_0'
epoch=190000

TYPE="query"
DATASET='_track'

python embed.py \
        --experiment_root ./experiments/VeRi/${expr_dir} \
        --dataset data/VeRi${DATASET}/VeRi_${TYPE}.csv \
        --filename VeRi_${TYPE}_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --flip_augment \
        --aggregator mean \
        --batch_size 128
#         --crop_augment five \

TYPE='test'

python embed.py \
        --experiment_root ./experiments/VeRi/${expr_dir} \
        --dataset data/VeRi${DATASET}/VeRi_${TYPE}.csv \
        --filename VeRi_${TYPE}_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --flip_augment \
        --aggregator mean \
        --batch_size 128
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
