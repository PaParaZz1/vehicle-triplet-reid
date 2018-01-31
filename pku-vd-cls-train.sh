#!/bin/sh
#
# This file calls train.py with all hyperparameters as for the TriNet
# experiment on market1501 in the original paper.

# if [ "$#" -lt 3 ]; then
#     echo "Usage: $0 PATH_TO_IMAGES RESNET_CHECKPOINT_FILE EXPERIMENT_ROOT ..."
#     echo "See the README for more info"
#     echo "Download ResNet-50 checkpoint from https://github.com/tensorflow/models/tree/master/research/slim#pre-trained-models"
#     exit 1
# fi

# Shift the arguments so that we can just forward the remainder.
# IMAGE_ROOT=$1 ; shift
# INIT_CHECKPT=$1 ; shift
# EXP_ROOT=$1 ; shift

IMAGE_ROOT=/home/qwang/Dataset/PKU-VD/VD1/ ; shift
INIT_CHECKPT=./pretrained_models/resnet_v2_50.ckpt ; shift
# INIT_CHECKPT=./pretrained_models/inception_v4.ckpt ; shift
EXP_ROOT=./experiments/pku-vd/expr_cls_supervised_0.01_cosine_resnet-50 ; shift


python train.py \
    --train_set data/pku-vd/VD1_train_cls.csv \
    --model_name resnet_v2_50 \
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
    --metric cosine \
    --loss batch_hard \
    --head_name fc1024_cls_supervised \
    --learning_rate 1e-4 \
    --train_iterations 400000 \
    --decay_start_iteration 10000 \
    --cls_loss_weight 0.01 \
    --lr_decay_factor 0.96 \
    --lr_decay_steps 4000 \
    --weight_decay_factor 0.0002 \
    --detailed_logs \
    "$@"
    # --resume \
