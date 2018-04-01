#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
source venv/bin/activate
expr_dir='expr_attention_euclidean_fc1024_inception_scratch'
epoch=20000

TYPE="query"

# python embed.py \
python embed_viz.py \
        --experiment_root ./experiments/VeRi/${expr_dir} \
        --dataset data/VeRi/VeRi_${TYPE}.csv \
        --filename VeRi_${TYPE}_${epoch}_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128
