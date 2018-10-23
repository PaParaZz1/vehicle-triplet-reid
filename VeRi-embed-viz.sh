#!/bin/sh

export CUDA_VISIBLE_DEVICES=5
source venv/bin/activate

# expr_dir='expr_attention_euclidean_fc1024_MBA_5b_concat_resnet_v2_50_0.05_0'
# epoch=70000
expr_dir='expr_attention_euclidean_fc1024_MBA_5b_kl_concat_resnet_v2_50_0.1_0'
epoch=50000

TYPE="query"

python embed_viz.py \
        --experiment_root ./experiments/VeRi/${expr_dir} \
        --dataset data/VeRi_track/VeRi_${TYPE}.csv \
        --filename VeRi_${TYPE}_${epoch}_viz_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128
