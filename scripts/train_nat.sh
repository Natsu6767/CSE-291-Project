#!/bin/bash

ALGO=sacv2_3d
TASK=$1
WORK_DIR=$2
SEED=$3
TRAIN_STEPS=$4
RL=$5
THREED=$6
UPDATE_3D_FREQ=$7
BSIZE=$8
RLENC=$9
PCONV=${10}
DENC=${11}
LR=${12}

python src/train.py \
    --algorithm $ALGO \
  	--task_name $TASK \
	--num_shared_layers 4 \
	--projection_dim 64 \
	--frame_stack 1 \
	--save_video \
	--exp_suffix $WORK_DIR \
    --seed $SEED \
	--train_steps $TRAIN_STEPS \
	--log_dir logs \
	--image_size 64 \
	--train_rl $RL \
	--train_3d $THREED \
	--init_steps 1000 \
	--prop_to_3d 1 \
	--bottleneck 16 \
	--update_3d_freq $UPDATE_3D_FREQ \
	--bsize_3d $BSIZE \
    --rl_enc $RLENC \
	--project_conv $PCONV \
	--double_enc $DENC \
	--lr_3d $LR \
	--lr_3dc $LR \
	--lr_3dp $LR

