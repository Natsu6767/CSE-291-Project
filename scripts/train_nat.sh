#!/bin/bash

ALGO=sacv2_3d
TASK=$1
WORK_DIR=$2
SEED=$3
TRAIN_STEPS=$4
RL=$5
THREED=$6
IMPALA=$7
LATENT=$8
PCONV=$9
ASPACE=${10}

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
	--prop_to_3d 0 \
	--bottleneck 16 \
    --use_impala $IMPALA \
	--use_latent $LATENT \
	--project_conv $PCONV \
	--action_space $ASPACE

