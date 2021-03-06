#!/bin/sh
#
# This file calls train.py with all hyperparameters as for the TriNet
# experiment on market1501 in the original paper.

# Shift the arguments so that we can just forward the remainder.
export CUDA_VISIBLE_DEVICES=5

METRIC='euclidean'
HEADS='fc1024_inception_multi-head_residual_attention'
BACKBONE='inception'
EXPR_NAME='_0'

IMAGE_ROOT=/data2/wangq/VD1/ ; shift
INIT_CHECKPT=./pretrained_models/inception_v4.ckpt ; shift
EXP_ROOT=./experiments/pku-vd/expr_attention_${METRIC}_${HEADS}_${BACKBONE}${EXPR_NAME} ; shift

python train.py \
    --train_set data/pku-vd/VD1_train.csv \
    --model_name ${BACKBONE} \
    --image_root $IMAGE_ROOT \
    --initial_checkpoint $INIT_CHECKPT \
    --experiment_root $EXP_ROOT \
    --flip_augment \
    --crop_augment \
    --detailed_logs \
    --embedding_dim 128 \
    --batch_p 18 \
    --batch_k 4 \
    --pre_crop_height 300 --pre_crop_width 300 \
    --net_input_height 224 --net_input_width 224 \
    --margin soft \
    --metric ${METRIC} \
    --loss batch_hard \
    --head_name ${HEADS} \
    --learning_rate 1e-4 \
    --train_iterations 400000 \
    --decay_start_iteration 10000 \
    --lr_decay_factor 0.96 \
    --lr_decay_steps 4000 \
    --weight_decay_factor 0.0002 \
    "$@"
    # --resume \
