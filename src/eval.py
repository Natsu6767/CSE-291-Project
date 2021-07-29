import torch
import torchvision
import os
import numpy as np
import gym
import utils
from copy import deepcopy
#from tqdm import tqdm
from arguments import parse_args
from env.wrappers import make_env
from algorithms.factory import make_agent
from video import VideoRecorder
import augmentations
import cv2


# def evaluate(env, agent, video, num_episodes, eval_mode, adapt=False):
# 	episode_rewards = []
# 	for i in tqdm(range(num_episodes)):
# 		if adapt:
# 			ep_agent = deepcopy(agent)
# 			ep_agent.init_pad_optimizer()
# 		else:
# 			ep_agent = agent
# 		obs = env.reset()
# 		video.init(enabled=True)
# 		done = False
# 		episode_reward = 0
# 		while not done:
# 			with utils.eval_mode(ep_agent):
# 				action = ep_agent.select_action(obs)
# 			next_obs, reward, done, _ = env.step(action)
# 			video.record(env, eval_mode)
# 			episode_reward += reward
# 			if adapt:
# 				ep_agent.update_inverse_dynamics(*augmentations.prepare_pad_batch(obs, next_obs, action))
# 			obs = next_obs

# 		video.save(f'eval_{eval_mode}_{i}.mp4')
# 		episode_rewards.append(episode_reward)

# 	return np.mean(episode_rewards)

def evaluate(env, agent, video, num_episodes, eval_mode, image_size, test_env=True):
	episode_rewards = []
	success_rate = []

	for i in range(num_episodes):
		obs = env.reset()
		video.init(enabled=True)
		done = False
		episode_reward = 0
		while not done:
			with utils.eval_mode(agent):
				action = agent.select_action(obs)
			#action = np.random.rand(2)
			obs, reward, done, info = env.step(action)
			#cv2.imwrite(str(i)+'fp.png', obs[3:6, :, :].transpose(1, 2, 0))
			#print("Obs received", obs.shape)
			video.record(env)
			episode_reward += reward
		if 'is_success' in info:
			success = float(info['is_success'])
			success_rate.append(success)

		_test_env = '_test_env'		
		video.save(f'{i}{_test_env}.mp4')

		episode_rewards.append(episode_reward)

	return np.nanmean(episode_rewards), np.nanmean(success_rate)


def main(args):

	seed_rewards = []
	for s in range(args.num_seeds):
		# Set seed
		utils.set_seed_everywhere(args.seed + s)
		if args.cameras==0:
			cameras=['third_person']
		elif args.cameras==1:
			cameras=['first_person']
		elif args.cameras==2:
			cameras = ['third_person', 'first_person']

		# Initialize environments
		gym.logger.set_level(40)
		env = make_env(
			domain_name=args.domain_name,
			task_name=args.task_name,
			seed=args.seed+s+42,
			episode_length=args.episode_length,
			n_substeps=args.n_substeps,
			frame_stack=args.frame_stack,
			action_repeat=args.action_repeat,
			image_size=args.image_size,
			mode=args.eval_mode,
			cameras=cameras, #['third_person', 'first_person']
			render=args.render, # Only render if observation type is state
			observation_type=args.observation_type,
			camera_dropout=args.camera_dropout
		)


		# Set working directory
		work_dir = os.path.join(args.log_dir, args.domain_name+'_'+args.task_name, args.algorithm, args.exp_suffix, str(args.seed))
		print('Working directory:', work_dir)
		assert os.path.exists(work_dir), 'specified working directory does not exist'
		model_dir = utils.make_dir(os.path.join(work_dir, 'model'))
		video_dir = utils.make_dir(os.path.join(work_dir, 'video'))
		video = VideoRecorder(video_dir if args.save_video else None, height=448, width=448)

		# Check if evaluation has already been run
		results_fp = os.path.join(work_dir, args.eval_mode+'.pt')
		assert not os.path.exists(results_fp), f'{args.eval_mode} results already exist for {work_dir}'

		# Prepare agent
		assert torch.cuda.is_available(), 'must have cuda enabled'
		cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
		print('Observations:', env.observation_space.shape)
		print('Cropped observations:', cropped_obs_shape)
		agent = make_agent(
			obs_shape=cropped_obs_shape,
			action_shape=env.action_space.shape,
			args=args
		)
		agent = torch.load(os.path.join(model_dir, str(args.train_steps)+'.pt'))
		agent.train(False)

		print(f'\nEvaluating {work_dir} for {args.eval_episodes} episodes (mode: {args.eval_mode})')
		reward, success_rate = evaluate(env, agent, video, args.eval_episodes, args.eval_mode, args.image_size)
		print('Reward:', int(reward))
		print('Success Rate:', int(success_rate))
		seed_rewards.append(int(reward))

		# # Save results
		# torch.save({
		# 	'args': args,
		# 	'reward': reward,
		# 	'adapt_reward': adapt_reward
		# }, results_fp)
		# print('Saved results to', results_fp)

	print('Average Reward over all the seeds:', int(np.nanmean(seed_rewards)))


if __name__ == '__main__':
	args = parse_args()
	main(args)
