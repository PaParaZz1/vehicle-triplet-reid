#!/bin/sh

export CUDA_VISIBLE_DEVICES=0,1,2,3
source ./venv/bin/activate

METRIC='euclidean'
# HEADS='fc1024_MBA_5b_kl_addition'
# HEADS='fc1024_MBA_5b_js_addition'
# HEADS='fc1024_MBA_5b_js_concat'
HEADS='fc1024_inception_MBA_5b_addition'
# BACKBONE='resnet_v2_50'
BACKBONE='inception'
EXPR_NAME='_0.01_0'

IMAGE_ROOT=/data/wangq/VeRi-776/ ; shift
# INIT_CHECKPT=./pretrained_models/resnet_v2_50.ckpt ; shift
INIT_CHECKPT=./pretrained_models/inception_v4.ckpt ; shift
EXP_ROOT=./experiments/VeRi/expr_attention_${METRIC}_${HEADS}_${BACKBONE}${EXPR_NAME} ; shift

python train_mba_multi_gpu.py \
    --initial_checkpoint $INIT_CHECKPT \
    --train_set data/VeRi_track/VeRi_train.csv \
    --model_name ${BACKBONE} \
    --image_root ${IMAGE_ROOT} \
    --experiment_root ${EXP_ROOT} \
    --flip_augment \
    --embedding_dim 128 \
    --batch_p 32 \
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
    # --resume \
    # --crop_augment \
    # --detailed_logs \
