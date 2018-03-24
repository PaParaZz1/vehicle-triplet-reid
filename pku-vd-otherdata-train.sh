#!/bin/sh
#
# This file calls train.py with all hyperparameters as for the TriNet
# experiment on market1501 in the original paper.

# Shift the arguments so that we can just forward the remainder.
export CUDA_VISIBLE_DEVICES=0

METRIC='euclidean'
HEADS='fc1024_inception_multi-residual-head_attention_5_branch'
BACKBONE='inception'
TRAIN_PART='total'
EXPR_NAME=''
DATASET='VD2'

IMAGE_ROOT=/data2/wangq/${DATASET}/ ; shift
if [ ${BACKBONE} == 'resnet_v2_50' ]; then
    INIT_CHECKPT=./experiments/pku-vd/pku-vd_resnet50_v2_results/checkpoint-360000 ; shift
    echo 'Finetune using resnet_v2_50 bachbone'
elif [ ${BACKBONE} == 'inception' ]; then
    INIT_CHECKPT=./experiments/pku-vd/ckpt_inception_v4/checkpoint-285886 ; shift
    echo 'Finetune using inception_v4 backbone'
else
    echo 'Wrong Backbone Networks name'
    exit 9
fi

# INIT_CHECKPT=./pretrained_models/inception_v4.ckpt

if [ ${TRAIN_PART} == 'head' ]; then
    PROCESSOR='train_finetune.py'
    LEARNING_RATE=1e-4
    EXPR_NAME=${EXPR_NAME} + '_fixed_finetune_0'
    echo 'Only finetune head'
elif [ ${TRAIN_PART} == 'total' ]; then
    PROCESSOR='train.py'
    LEARNING_RATE=1e-5
    EXPR_NAME=${EXPR_NAME} + '_finetune_0'
    echo 'Finetune the whole networks'
else
    echo 'Wrong training part'
    exit 9
fi

EXPR_NAME=${EXPR_NAME} + '_finetune_0'

python ${PROCESSOR} \
    --train_set data/pku-vd/${DATASET}_train.csv \
    --model_name inception \
    --image_root $IMAGE_ROOT \
    --initial_checkpoint ./pretrained_models/inception_v4.ckpt \
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
    --metric euclidean \
    --loss batch_hard \
    --head_name fc1024 \
    --learning_rate 1e-4 \
    --train_iterations 400000 \
    --decay_start_iteration 0 \
    --lr_decay_factor 0.96 \
    --lr_decay_steps 4000 \
    --weight_decay_factor 0.0002 \
    "$@"
    # --resume \
