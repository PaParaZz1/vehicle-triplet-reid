#!/bin/sh
#
# This file calls train.py with all hyperparameters as for the TriNet
# experiment on market1501 in the original paper.

# Shift the arguments so that we can just forward the remainder.
export CUDA_VISIBLE_DEVICES=0,1,2,4

METRIC='euclidean'
HEADS='fc1024_MBA_5b_addition'
BACKBONE='resnet_v2_50'
EXPR_NAME='_0'

IMAGE_ROOT=/data2/wangq/VD1/ ; shift
INIT_CHECKPT=./pretrained_models/resnet_v2_50.ckpt ; shift
EXP_ROOT=./experiments/pku-vd/expr_attention_${METRIC}_${HEADS}_${BACKBONE}${EXPR_NAME} ; shift

python train_multi_gpu.py \
    --train_set data/pku-vd/VD1_train.csv \
    --model_name ${BACKBONE} \
    --image_root $IMAGE_ROOT \
    --experiment_root $EXP_ROOT \
    --flip_augment \
    --embedding_dim 128 \
    --batch_p 20 \
    --batch_k 4 \
    --pre_crop_height 224 --pre_crop_width 224 \
    --net_input_height 224 --net_input_width 224 \
    --margin soft \
    --metric ${METRIC} \
    --loss batch_hard \
    --head_name ${HEADS} \
    --learning_rate 1e-4 \
    --train_iterations 400000 \
    --decay_start_iteration 0 \
    --lr_decay_factor 0.96 \
    --lr_decay_steps 4000 \
    --weight_decay_factor 0.0002 \
    "$@"
    # --detailed_logs \
    # --crop_augment \
    # --resume \
    # --initial_checkpoint $INIT_CHECKPT \
