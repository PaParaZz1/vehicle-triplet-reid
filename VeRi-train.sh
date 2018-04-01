#!/bin/sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
source ./venv/bin/activate

# IMAGE_ROOT=/home/qwang/Dataset/VeRi/ ; shift
IMAGE_ROOT=/data2/wangq/VeRi-776/ ; shift
METRIC='euclidean'
# HEADS='fc1024_inception_MBA_5b_addition'
HEADS='fc1024'
BACKBONE='inception'
EXPR_NAME='_scratch'
INIT_CHECKPT=./pretrained_models/inception_v4.ckpt ; shift
EXP_ROOT=./experiments/VeRi/expr_attention_${METRIC}_${HEADS}_${BACKBONE}${EXPR_NAME} ; shift


python train_multi_gpu.py \
    --train_set data/VeRi/VeRi_train.csv \
    --model_name inception \
    --image_root $IMAGE_ROOT \
    --initial_checkpoint $INIT_CHECKPT \
    --experiment_root $EXP_ROOT \
    --flip_augment \
    --embedding_dim 128 \
    --batch_p 20 \
    --batch_k 4 \
    --pre_crop_height 224 --pre_crop_width 224 \
    --net_input_height 224 --net_input_width 224 \
    --margin soft \
    --metric euclidean \
    --loss batch_hard \
    --learning_rate 1e-4 \
    --train_iterations 400000 \
    --decay_start_iteration 10000 \
    --lr_decay_factor 0.96 \
    --lr_decay_steps 4000 \
    --weight_decay_factor 0.0002 \
    --resume \
    "$@"
    # --crop_augment \
    # --detailed_logs \
