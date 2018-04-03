#!/bin/sh
#
# This file calls train.py with all hyperparameters as for the TriNet
# experiment on market1501 in the original paper.

# Shift the arguments so that we can just forward the remainder.
export CUDA_VISIBLE_DEVICES=2,3

source ./venv/bin/activate

METRIC='euclidean'
HEADS='fc1024_inception_MBA_5b_addition'
BACKBONE='inception'
TRAIN_PART='total'
EXPR_NAME='_weakly_detection_2'

IMAGE_ROOT=/data2/wangq/VD1/ ; shift
INIT_CHECKPT=./experiments/pku-vd/ckpt_inception_v4/checkpoint-285886 ; shift

EXP_ROOT=./experiments/pku-vd/expr_attention_${METRIC}_${HEADS}_${BACKBONE}${EXPR_NAME} ; shift
LEARNING_RATE=1e-4

python train_weakly_detection_multi_gpu.py \
    --train_set data/pku-vd/VD1_train_cls.csv \
    --model_name ${BACKBONE} \
    --image_root $IMAGE_ROOT \
    --initial_checkpoint $INIT_CHECKPT \
    --experiment_root $EXP_ROOT \
    --flip_augment \
    --crop_augment \
    --embedding_dim 128 \
    --batch_p 8 \
    --batch_k 16 \
    --pre_crop_height 224 --pre_crop_width 224 \
    --net_input_height 224 --net_input_width 224 \
    --margin soft \
    --metric ${METRIC} \
    --loss batch_hard \
    --head_name ${HEADS} \
    --learning_rate ${LEARNING_RATE} \
    --train_iterations 400000 \
    --decay_start_iteration 0 \
    --lr_decay_factor 0.96 \
    --lr_decay_steps 4000 \
    --weight_decay_factor 0.0002 \
    "$@"
    # --resume \
    # --detailed_logs \
