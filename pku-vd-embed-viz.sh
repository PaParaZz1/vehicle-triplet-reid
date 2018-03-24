#!/bin/sh

export CUDA_VISIBLE_DEVICES=5
source ./venv/bin/activate

epoch=300000
dataset_size=''
# expr_dir='expr_attention_euclidean_fc1024_inception_multi-residual-head_attention_8_branch_inception__finetune_0'
# expr_dir='expr_attention_euclidean_fc1024_inception_multi-residual-head_attention_5_branch_inception_10.0_finetune_0'
expr_dir='expr_attention_euclidean_fc1024_inception_multi-residual-head_attention_5_branch_inception'

# python embed_mixed_viz.py \
python embed_viz.py \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --dataset data/pku-vd/slt_img.csv \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings_viz.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128
        # --dataset data/pku-vd/VD1_${dataset_size}query.csv \

