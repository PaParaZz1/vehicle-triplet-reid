#!/bin/sh
#
# This file calls train.py with all hyperparameters as for the TriNet
# experiment on market1501 in the original paper.

# Shift the arguments so that we can just forward the remainder.
export CUDA_VISIBLE_DEVICES=4

METRIC='euclidean'
# HEADS='fc1024_inception_mixed_attention'
HEADS='fc1024'
BACKBONE='inception'
LEARNING_RATE=1e-5
PROCESSOR='train_rl.py'
EXPR_NAME='_finetune_rl_0'
INIT_CHECKPT=./experiments/pku-vd/ckpt_inception_v4/checkpoint-285886 ; shift

EXP_ROOT=./experiments/pku-vd/expr_attention_${METRIC}_${HEADS}_${BACKBONE}${EXPR_NAME} ; shift
IMAGE_ROOT=/data2/wangq/VD1/ ; shift

# if [ ${BACKBONE} == 'resnet_v2_50' ]; then
#     INIT_CHECKPT=./experiments/pku-vd/pku-vd_resnet50_v2_results/checkpoint-360000 ; shift
#     echo 'Finetune using resnet_v2_50 bachbone'
# elif [ ${BACKBONE} == 'inception' ]; then
#     INIT_CHECKPT=./experiments/pku-vd/ckpt_inception_v4/checkpoint-285886 ; shift
#     echo 'Finetune using inception_v4 backbone'
# else
#     echo 'Wrong Backbone Networks name'
#     exit 9
# fi

python ${PROCESSOR} \
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
    --learning_rate ${LEARNING_RATE} \
    --train_iterations 400000 \
    --decay_start_iteration 0 \
    --lr_decay_factor 0.96 \
    --lr_decay_steps 4000 \
    --weight_decay_factor 0.0002 \
    "$@"
    # --resume \
