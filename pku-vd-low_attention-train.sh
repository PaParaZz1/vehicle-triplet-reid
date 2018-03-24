#!/bin/sh
export CUDA_VISIBLE_DEVICES=4

METRIC='euclidean'
HEADS='fc1024'
BACKBONE='inception_attention'
TRAIN_PART='total'
EXPR_NAME='low-level_attention'

INIT_CHECKPT=./pretrained_models/inception_v4.ckpt ; shift

IMAGE_ROOT=/data2/wangq/VD1/ ; shift

PROCESSOR='train.py'
LEARNING_RATE=1e-4

EXP_ROOT=./experiments/pku-vd/expr_attention_${METRIC}_${HEADS}_${BACKBONE}${EXPR_NAME} ; shift

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
    --resume \
    "$@"
