#!/bin/sh

export CUDA_VISIBLE_DEVICES=7
source ./venv/bin/activate

dataset_size=''
# epoch=430000
# expr_dir='expr_attention_euclidean_fc1024_inception_MBA_5b_addition_inception_1.0_1_finetune_0'
# epoch=370000
# expr_dir='expr_attention_euclidean_fc1024_inception_multi-residual-head_attention_5_branch_inception_10.0_finetune_0'
# epoch=300000
# expr_dir='expr_attention_euclidean_fc1024_inception_multi-residual-head_attention_5_branch_inception'
epoch=280000
expr_dir='expr_attention_euclidean_fc1024_inception_multi-residual-head_attention_8_branch_inception__finetune_0'

# python embed_mixed_viz.py \
python embed_viz.py \
        --dataset data/pku-vd/VD1_${dataset_size}query.csv \
        --experiment_root ./experiments/pku-vd/${expr_dir} \
        --filename pku-vd_VD1_${dataset_size}query_${epoch}_embeddings_viz.h5 \
        --checkpoint checkpoint-${epoch} \
        --batch_size 128
        # --dataset data/pku-vd/slt_img.csv \
