export MUJOCO_GL=egl
CUDA_VISIBLE_DEVICES=1 python3 src/test_simulation.py \
  --algorithm drq \
  --domain_name robot \
  --task_name pickplace \
  --save_video \
  --episode_length 50 \
  --exp_suffix dev \
  --eval_mode none \
  --eval_freq 1 \
  --train_steps 20k \
  --frame_stack 1 \
  --eval_episodes 1 \
  --image_size 100 \
  --image_crop_size 84 \
  --seed 0 \
  --batch_size 12 \
  --from_state \
  --cameras 0 \
  --camera_dropout 0 \
  --num_head_layers 0 \
  # --render True
