import os
import numpy as np
import gym
import utils
import time
from arguments import parse_args
from env.wrappers import make_env
from logger import Logger
import matplotlib.pyplot as plt
import torch
import cv2
from algorithms.factory import make_agent

def main(args):
	utils.set_seed_everywhere(args.seed)
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
		seed=args.seed,
		episode_length=args.episode_length,
		n_substeps=args.n_substeps,
		frame_stack=args.frame_stack,
		action_repeat=args.action_repeat,
		image_size=args.image_size,
		cameras=cameras, #['third_person', 'first_person']
		mode='train',
		render=args.render,
		action_space ='xy',
		observation_type= 'state' if args.from_state else 'image',
		)

	if args.render: env.render()

	num_episodes = 10000

	print("Observatins", env.observation_space.shape)

	# assert torch.cuda.is_available(), 'must have cuda enabled'
	# replay_buffer = utils.ReplayBuffer(
	# 	obs_shape=env.observation_space.shape,
	# 	action_shape=env.action_space.shape,
	# 	capacity=args.train_steps,
	# 	batch_size=args.batch_size
	# )

	# cropped_obs_shape = (3*args.frame_stack, args.image_crop_size, args.image_crop_size)
	# print('Observations:', env.observation_space.shape)
	# print('Cropped observations:', cropped_obs_shape)

	# agent = make_agent(
	# 	obs_shape=cropped_obs_shape,
	# 	action_shape=env.action_space.shape,
	# 	args=args
	# )

	# from torchsummary import summary
	# summary(agent.actor, env.observation_space.shape)

	env.render()
	success_rate = []
	obs = env.reset()
	for i in range(num_episodes):
		
		done = False
		episode_reward = 0

		action = np.array([0.1, 0.1])
		obs, reward, done, info = env.step(action)
		episode_reward += reward
		if args.render: env.render()
		
		#obs = np.concatenate((obs['third_person'], obs['first_person']), axis=0)
		#replay_buffer.add(obs, action, reward, obs, done)
		#action = agent.sample_action(obs)
		print("action sampled", action)
		# if i%50==0:
		# 	env.reset()
		env.reset()
		#obs2, action2, reward2, obs2, done2 = replay_buffer.sample(n=1)
		env.render()
		#print("REading buffer ", obs2.shape)
		#assert (obs2==obs).all()
		#assert (action2==action).all()
		#cv2.imwrite(str(i)+'azure__.png', obs.transpose(1, 2, 0))
		

if __name__ == '__main__':
	args = parse_args()
	main(args)
