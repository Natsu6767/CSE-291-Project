#!/bin/bash

ALGO=sacv2_3d
TASK=$1
WORK_DIR=$2
SEED=$3
TRAIN_STEPS=$4
RL=$5
THREED=$6
HDIM=$7
RLENC=$8
CAM=$9

python main.py \
    --algorithm $ALGO \
  	--task_name $TASK \
	--num_shared_layers 4 \
	--projection_dim 64 \
	--frame_stack 1 \
	--save_video \
	--exp_suffix $WORK_DIR \
    --seed $SEED
	--train_steps $TRAIN_STEPS \
	--log_dir logs \
	--image_size 64 \
	--train_rl $RL \
	--train_3d $THREED \
	--init_steps 1000 \
	--prop_to_3d 1 \
	--bottleneck 16 \
    --hidden_dim $HDIM \
    --rl_enc $RLENC \
    --cameras $CAM

