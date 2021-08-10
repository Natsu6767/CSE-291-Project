import torch
import os
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from logger import Logger
from video import VideoRecorder
from torch.utils.tensorboard import SummaryWriter

os.environ["IMAGEIO_FFMPEG_EXE"] = "/usr/bin/ffmpeg"
torch.backends.cudnn.benchmark = True


def evaluate(env, agent, video, num_episodes, L, step, test_env=False):
	episode_rewards = []
	success_rate = []
	for i in range(num_episodes):
		obs = env.reset()
		video.init(enabled=(i==0))
		done = False
		episode_reward = 0
		while not done:
			with torch.no_grad(), utils.eval_mode(agent):
				action = agent.select_action(obs)
			obs, reward, done, info = env.step(action)
			#video.record(env)
			episode_reward += reward
		if 'is_success' in info:
			success = float(info['is_success'])
			success_rate.append(success)

		if L is not None:
			_test_env = '_test_env' if test_env else ''
			#video.save(f'{step}{_test_env}.mp4')
			L.log(f'eval/episode_reward{_test_env}', episode_reward, step)
			if 'is_success' in info:
				L.log(f'eval/sucess_rate', success, step)
		episode_rewards.append(episode_reward)

	return np.nanmean(episode_rewards), np.nanmean(success_rate)


def visualize_configurations(env, args):
	from torchvision.utils import make_grid, save_image
	frames = []
	for i in range(20):
		env.reset()
		frame = torch.from_numpy(env.render_obs(mode='rgb_array', height=448, width=448, camera_id=0).copy()).permute(2,0,1).float().div(255)
		frames.append(frame)
	save_image(make_grid(torch.stack(frames), nrow=5), f'{args.domain_name}_{args.task_name}.png')


def main(args):
	# Set seed
	utils.set_seed_everywhere(args.seed)
	if args.cameras == 0:
		cameras="dynamic"
	elif args.cameras == 1:
		cameras="dynamic_2"
	else:
		print("ERRORR")
		cameras = None

	# Initialize environments
	gym.logger.set_level(40)
	env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed,
		episode_length=args.episode_length,
		n_substeps=args.n_substeps,
		frame_stack=args.frame_stack,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode='train',
		cameras=cameras, #['third_person', 'first_person']
		render=args.render, # Only render if observation type is state
		camera_dropout=args.camera_dropout,
		observation_type='state' if args.from_state else 'image',
		action_space=args.action_space
	)
	test_env = make_env(
		domain_name=args.domain_name,
		task_name=args.task_name,
		seed=args.seed+42,
		episode_length=args.episode_length,
		n_substeps=args.n_substeps,
		frame_stack=args.frame_stack,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		mode=args.eval_mode,
		cameras=cameras, #['third_person', 'first_person']
		render=args.render, # Only render if observation type is state
		camera_dropout=args.camera_dropout,
		observation_type='state' if args.from_state else 'image',
		action_space=args.action_space
	) if args.eval_mode is not None else None

	# Visualize initial configurations
	if args.visualize_configurations:
		visualize_configurations(env, args)

	# Create working directory
	work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, args.exp_suffix, str(args.seed))
	print('Working directory:', work_dir)
	assert not os.path.exists(os.path.join(work_dir, 'train.log')) or args.exp_suffix == 'dev', 'specified working directory already exists'
	utils.make_dir(work_dir)
	model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
	video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
	video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448, fps=15 if args.domain_name == 'robot' else 25)
	utils.write_info(args, os.path.join(work_dir, 'info.log'))

	# Prepare agent
	assert torch.cuda.is_available(), 'must have cuda enabled'
	replay_buffer = utils.ReplayBuffer(
		obs_shape=env.observation_space.shape,
		action_shape=env.action_space.shape,
		capacity=args.train_steps,
		batch_size=args.batch_size
	)

	print('Observations:', env.observation_space.shape)
	print('Action space:', f'{args.action_space} ({env.action_space.shape[0]})')

	if args.use_latent:
		a_obs_shape = (args.bottleneck*32, 32, 32)
	elif args.use_impala:
		a_obs_shape = (128, 8, 8)
	else:
		a_obs_shape = (32, 26, 26)

	agent = make_agent(
		obs_shape=a_obs_shape,
		action_shape=env.action_space.shape,
		args=args
	)

	start_step, episode, episode_reward, info, done, episode_success = 0, 0, 0, {}, True, 0
	L = Logger(work_dir)
	writer = SummaryWriter(log_dir=os.path.join(work_dir, "tboard"))

	start_time = time.time()
	training_time = start_time
	video_tensor = list()
	for step in range(start_step, args.train_steps+1):
		if done:
			if step > start_step:
				L.log('train/duration', time.time() - start_time, step)
				start_time = time.time()
				if step % args.log_train_video == 0:
					t_vid_tensor = torch.tensor(video_tensor).unsqueeze(0).float().div(255)
					writer.add_video("Training Video", t_vid_tensor, step)
				L.dump(step)

			# Evaluate agent periodically
			if step % args.eval_freq == 0:
				print('Evaluating:', work_dir)
				L.log('eval/episode', episode, step)
				evaluate(env, agent, video, args.eval_episodes, L, step)
				if test_env is not None:
					evaluate(test_env, agent, video, args.eval_episodes, L, step, test_env=True)
				L.dump(step)

				# Evaluate 3D
				if args.train_3d:
					obs = env.reset()
					# Execute one timestep to randomize the camera and environemnt.
					a_eval = env.action_space.sample()
					obs, _, _, _ = env.step(a_eval)
					# Select the camera views
					o1 = obs[:3]
					o2 = obs[3:]
					# Concatenate and convert to torch tensor and add unit batch dimensions
					images_rgb = np.concatenate([np.expand_dims(o1, axis=0),
												 np.expand_dims(o2, axis=0)], axis=0)
					images_rgb = torch.from_numpy(images_rgb).float().cuda().unsqueeze(0).div(255)
					agent.gen_interpolate(images_rgb, writer, step)

			# Save agent periodically
			if step == 100000 or step == args.train_steps:# step > start_step and step % args.save_freq == 0:
				torch.save(agent, os.path.join(model_dir, f'{step}.pt'))

			L.log('train/episode_reward', episode_reward, step)
			L.log('train/success_rate', episode_success/args.episode_length, step)

			writer.add_scalar('Episode Reward (Training)', episode_reward,step)
			writer.add_scalar('Success Rate (Training)', episode_success/args.episode_length, step)

			obs = env.reset()
			done = False

			video_tensor = list()
			video_tensor.append(obs[:3])
			episode_reward = 0
			episode_step = 0
			episode += 1
			episode_success = 0

			L.log('train/episode', episode, step)

		# Sample action and update agent
		if step < args.init_steps:
			action = env.action_space.sample()
		else:
			with torch.no_grad(), utils.eval_mode(agent):
				action = agent.sample_action(obs)
			num_updates = args.init_steps//args.update_freq if step == args.init_steps else 1
			for i in range(num_updates):
				agent.update(replay_buffer, L, writer, step)

		# Take step
		next_obs, reward, done, info = env.step(action)
		replay_buffer.add(obs, action, reward, next_obs)
		episode_reward += reward
		obs = next_obs
		video_tensor.append(obs[:3])
		episode_success += float(info['is_success'])
		episode_step += 1
	print('Completed training for', work_dir)
	print("Total Training Time: ", round((time.time() - training_time) / 3600, 2), "hrs")


if __name__ == '__main__':
	args = parse_args()
	main(args)
