#!/bin/sh

export CUDA_VISIBLE_DEVICES=7
source ./venv/bin/activate

dataset_size=''
epoch=300000
# expr_dir='ckpt_inception_v4'
expr_dir='expr_attention_euclidean_fc1024_inception_mixed_attention_inception_finetune_0'
# expr_dir='expr_attention_euclidean_fc1024_inception_multi-residual-head_attention_8_branch_inception__finetune_0'


python embed_mixed_viz.py \
        --dataset data/pku-vd/slt_img.csv \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings_viz.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 10
