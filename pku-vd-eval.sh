#!/bin/sh

epoch=20000
dataset_size='small_'
# expr_dir='expr_cls_euclidean_1.0_resnet-50'
# expr_dir='expr_cls_euclidean_1e-3_balanced_resnet-50'
# expr_dir='expr_cls_euclidean_1e-2_balanced_resnet-50'
expr_dir='expr_cls_euclidean_1.0_fc1024_cls_projection_resnet_v2_50_0'
# expr_dir='expr_cls_euclidean_0.5_fc1024_cls_projection_resnet_v2_50'
# expr_dir='expr_cls_euclidean_0.5_fc1024_cls_projection_resnet_v2_50__2'
# expr_dir='expr_cls_euclidean_1.0_fc1024_cls_wo_projection_resnet_v2_50'

python ./evaluate.py \
    --excluder diagonal \
    --query_dataset ./data/pku-vd/VD1_query.csv \
    --query_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_query_${epoch}_embeddings.h5 \
    --gallery_dataset ./data/pku-vd/VD1_${dataset_size}query.csv \
    --gallery_embeddings ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_embeddings.h5 \
    --filename ./experiments/pku-vd/${expr_dir}/pku-vd_VD1_${dataset_size}query_${epoch}_evaluation.json \
    --batch_size 64 \
    --metric euclidean \
    # --metric euclidean \
