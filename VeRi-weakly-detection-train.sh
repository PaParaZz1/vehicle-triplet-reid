#!/bin/sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
source ./venv/bin/activate

METRIC='euclidean'
# HEADS='fc1024'
HEADS='fc1024_inception_MBA_5b_addition_joint'
BACKBONE='inception'
TRAIN_PART='total'
EXPR_NAME='_scratch_1'

IMAGE_ROOT=/data2/wangq/VeRi-776/ ; shift
INIT_CHECKPT=./pretrained_models/inception_v4.ckpt ; shift
# INIT_CHECKPT=./experiments/VeRi/ckpt_VeRi_inception_scratch/checkpoint-300000 ; shift

EXP_ROOT=./experiments/VeRi/expr_attention_${METRIC}_${HEADS}_${BACKBONE}${EXPR_NAME} ; shift
LEARNING_RATE=1e-4

python train_weakly_detection_multi_gpu.py \
    --train_set data/VeRi/VeRi_train.csv \
    --model_name ${BACKBONE} \
    --image_root $IMAGE_ROOT \
    --initial_checkpoint $INIT_CHECKPT \
    --experiment_root $EXP_ROOT \
    --flip_augment \
    --crop_augment \
    --embedding_dim 128 \
    --batch_p 20 \
    --batch_k 4 \
    --pre_crop_height 224 --pre_crop_width 224 \
    --net_input_height 224 --net_input_width 224 \
    --margin soft \
    --metric ${METRIC} \
    --loss joint_batch_hard \
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
