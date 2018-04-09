#!/bin/sh

export CUDA_VISIBLE_DEVICES=6
source venv/bin/activate

# expr_dir='expr_attention_euclidean_fc1024_inception_MBA_5b_addition_inception_finetune_2'
# epoch=50000
expr_dir='expr_attention_euclidean_fc1024_inception_MBA_5b_addition_inception_6'
epoch=260000

TYPE="query"

python embed_viz.py \
        --experiment_root ./experiments/VeRi/${expr_dir} \
        --dataset data/VeRi_track/VeRi_${TYPE}.csv \
        --filename VeRi_${TYPE}_${epoch}_viz_embeddings.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128
