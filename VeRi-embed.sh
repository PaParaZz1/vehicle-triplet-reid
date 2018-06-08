#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
source venv/bin/activate

# expr_dir='expr_attention_euclidean_fc1024_MBA_5b_addition_resnet_v2_50_0.1_1'
# epoch=100000

expr_dir='expr_attention_euclidean_fc1024_MBA_5b_addition_resnet_v2_50_0.0_1'
epoch=190000

# expr_dir='expr_attention_euclidean_fc1024_MBA_5b_addition_resnet_v2_50_0.01_1'
# epoch=50000

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
