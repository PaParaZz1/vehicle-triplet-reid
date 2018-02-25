#!/bin/sh

export CUDA_VISIBLE_DEVICES=6
epoch=0
dataset_size=''
# expr_dir='expr_attention_euclidean_fc1024_mixed_attention_resnet_v2_50_finetune'
# expr_dir='expr_attention_euclidean_fc1024_spatial_attention_softmax_resnet_v2_50_finetune'
expr_dir='expr_attention_euclidean_fc1024_inception_spatial_attention_inception_finetune_4'
# expr_dir='expr_attention_euclidean_fc1024_recurrent_attention_wstop_resnet_v2_50_finetune'
# expr_dir='expr_attention_euclidean_fc1024_recurrent_attention_resnet_v2_50_0'
# expr_dir='expr_attention_euclidean_fc1024_inception_spatial_attention_inception_0'
# expr_dir='expr_attention_euclidean_fc1024_spatial_attention_resnet_v2_50_0'

python embed.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128
        # --flip_augment \
        # --crop_augment five \
        # --aggregator mean

dataset_size='small_'

python embed.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128
        # --flip_augment \
        # --crop_augment five \
        # --aggregator mean

python ./evaluate.py \
    --excluder diagonal \
    --query_dataset ./data/pku-vd/VD1_query.csv \
    --query_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_query_${epoch}_embeddings.h5 \
    --gallery_dataset ./data/pku-vd/VD1_${dataset_size}query.csv \
    --gallery_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
    --metric euclidean \
    --filename ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_evaluation.json \
    --batch_size 64 \
    
