export MUJOCO_GL=egl
CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
	--algorithm sacv2_3d \
  --task_name reach \
	--num_shared_layers 4 \
	--projection_dim 64 \
	--frame_stack 1 \
	--save_video \
	--exp_suffix dev \
	--train_steps 100k \
	--log_dir logs \
	--seed 0 \
	--image_size 64 \
	--train_3d 1 \
	--train_rl 1 \
	--init_steps 1000 \
	--prop_to_3d 1 \
	--bottleneck 8
