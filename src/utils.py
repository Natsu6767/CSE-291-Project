import torch
import torchvision
from torch.utils.data import Dataset
import numpy as np
import os
import json
import random
import subprocess
import platform
from datetime import datetime


def array_init(capacity, num_views, dims, dtype):
    """Preallocate array in memory"""
    chunks = 20
    zero_dim_size = int(capacity / chunks)
    array = np.zeros((capacity, num_views, *dims), dtype=dtype)
    temp = np.ones((zero_dim_size, num_views, *dims), dtype=dtype)

    for i in range(chunks):
        array[i * zero_dim_size:(i + 1) * zero_dim_size] = temp

    return array




class eval_mode(object):
	def __init__(self, *models):
		self.models = models

	def __enter__(self):
		self.prev_states = []
		for model in self.models:
			self.prev_states.append(model.training)
			model.train(False)

	def __exit__(self, *args):
		for model, state in zip(self.models, self.prev_states):
			model.train(state)
		return False


def soft_update_params(net, target_net, tau):
	for param, target_param in zip(net.parameters(), target_net.parameters()):
		target_param.data.copy_(tau * param.data + (1 - tau) * target_param.data)


def cat(x, y, axis=0):
	return torch.cat([x, y], axis=0)


def set_seed_everywhere(seed):
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)
	np.random.seed(seed)
	random.seed(seed)


def write_info(args, fp):
	data = {
		'host': platform.node(),
		'cwd': os.getcwd(),
		'timestamp': str(datetime.now()),
		'git': subprocess.check_output(["git", "describe", "--always"]).strip().decode(),
		'args': vars(args)
	}
	with open(fp, 'w') as f:
		json.dump(data, f, indent=4, separators=(',', ': '))


def load_config(key=None):
	path = os.path.join('setup', 'config.cfg')
	with open(path) as f:
		data = json.load(f)
	if key is not None:
		return data[key]
	return data


def make_dir(dir_path):
	try:
		os.makedirs(dir_path)
	except OSError:
		pass
	return dir_path


def prefill_memory(capacity, obs_shape):
	obses = []
	if len(obs_shape) > 1:
		c,h,w = obs_shape
		for _ in range(capacity):
			frame = np.ones((3,h,w), dtype=np.uint8)
			obses.append(frame)
	else:
		for _ in range(capacity):
			obses.append(np.ones(obs_shape[0], dtype=np.float32))

	return obses


class ReplayBuffer(Dataset):
	"""Buffer to store environment transitions"""
	def __init__(self, obs_shape, action_shape, capacity, batch_size, n_step=3, episode_length=None, discount=None):
		self._capacity = capacity
		self._batch_size = batch_size
		self._n_step = n_step
		self._episode_length = episode_length
		self._discount = discount
		self._obses = prefill_memory(capacity, obs_shape)
		self._next_obses = [None]*capacity
		self._actions = np.empty((capacity, *action_shape), dtype=np.float32)
		self._rewards = np.empty((capacity, 1), dtype=np.float32)
		self._episodes = []
		self._current_episode = 0
		self._idx = 0
		self._full = False

	def __len__(self):
		return self._capacity if self._full else self._idx

	def __getitem__(self, idx):
		if self._n_step > 1:
			obs, next_obs = self._encode_obses((idx, idx + self._n_step - 1))
			reward = np.zeros((self._batch_size, 1), dtype=np.float32)
			discount = np.ones((self._batch_size, 1), dtype=np.float32)
			for i in range(self._n_step):
				reward += discount * self._rewards[idx + i]
				discount *= self._discount
			reward = torch.as_tensor(reward).cuda()
		else:
			obs, next_obs = self._encode_obses(idx)
			reward = torch.as_tensor(self._rewards[idx]).cuda()
		action = torch.as_tensor(self._actions[idx]).cuda()
		return obs, action, reward, next_obs

	def add(self, obs, action, reward, next_obs, episode):
		self._obses[self._idx] = obs
		self._next_obses[self._idx] = next_obs
		np.copyto(self._actions[self._idx], action)
		np.copyto(self._rewards[self._idx], reward)
		if episode > self._current_episode:
			self._episodes.append(self._idx)
			self._current_episode = episode
		self._idx = (self._idx + 1) % self._capacity
		self._full = self._full or self._idx == 0

	def _get_idx(self, n=None):
		if n is None:
			n = self._batch_size
		if self._n_step == 1:
			return np.random.randint(0, len(self), size=n)
		assert self._n_step in {3, 5}, 'n_step must be 3 or 5'
		assert len(self._episodes) > 1, 'need more than 1 episode before sampling with n-step returns'
		episodes = np.random.choice(self._episodes[:-1], size=n)
		steps = np.random.randint(0, self._episode_length-self._n_step+1, size=n)
		return episodes + steps

	def _encode_obses(self, idx):
		if not isinstance(idx, tuple):
			idx = (idx, idx)
		obses = torch.as_tensor(np.stack([self._obses[i] for i in idx[0]])).cuda().float()
		next_obses = torch.as_tensor(np.stack([self._next_obses[i] for i in idx[1]])).cuda().float()
		return obses, next_obses

	def sample_drq(self, n=None, pad=4):
		raise NotImplementedError('call sample() and apply aug in agent.update() instead')

	def sample(self, n=None):
		idx = self._get_idx(n)
		return self[idx]

class ReplayBuffer2(object):
    """Buffer to store environment transitions"""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, num_views=2):
        self.capacity = capacity
        self.batch_size = batch_size
        # Only use a single replay buffer. Reduce memory required.
        #if self.camera_obs:
        self.obs = array_init(capacity, num_views, obs_shape, dtype=np.uint8)
        #else:
        #    self.obs = array_init(capacity, obs_shape, dtype=np.float64)
        #    self.next_obs = array_init(capacity, obs_shape, dtype=np.float64)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.full = False
        #self.use_crop = use_crop_aug

    def add(self, obs, action, reward, next_obs, done):
        try:
            o1, o2 = obs
        except:
            import pdb;
            pdb.set_trace()

        obs_add = np.concatenate([np.expand_dims(o1, axis=0),
                                  np.expand_dims(o2, axis=0)], axis=0)
        np.copyto(self.obs[self.idx], obs_add)  # Add current obs

        np.copyto(self.actions[self.idx], action)  # Add action
        np.copyto(self.rewards[self.idx], reward)  # Add reward
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def _get_idxs(self, n=None):
        if n is None:
            n = self.batch_size

        assert self.idx > 1 or self.full, "Buffer needs at least 2 capacity filled"

        def sample_i():
            return np.random.randint(
                0, self.capacity if self.full else self.idx - 1, size=n)

        i = sample_i()
        if (self.full):
            while (self.idx == 0 and (i == self.capacity - 1).any()):
                i = sample_i()
            while ((i == self.idx - 1).any()):
                i = sample_i()

        return i

    def sample(self, n=None):
        idxs = self._get_idxs(n)

        obs = torch.from_numpy(self.obs[idxs]).float().div(255).cuda()
        actions = torch.as_tensor(self.actions[idxs])
        rewards = torch.as_tensor(self.rewards[idxs])

        nxt_o_idx = (idxs + 1) % self.capacity
        next_obs = torch.from_numpy(self.obs[nxt_o_idx]).float().div(255).cuda()[:, 0]

        not_dones = torch.as_tensor(self.not_dones[idxs])

        """b, v, c, h, w = obs.shape
        #if self.use_crop:
            obs = turbo_crop(obs.view(-1, c, h, w), size=128)
            n, c, h, w = obs.shape
            obs = obs.view(b, v, c, h, w)

        if self.use_crop:
            next_obs = turbo_crop(next_obs, size=128)"""

        return obs, actions.cuda(), rewards.cuda(), next_obs, not_dones.cuda()



class LazyFrames(object):
	def __init__(self, frames, extremely_lazy=True):
		self._frames = frames
		self._extremely_lazy = extremely_lazy
		self._out = None

	@property
	def frames(self):
		return self._frames

	def _force(self):
		if self._extremely_lazy:
			return np.concatenate(self._frames, axis=0)
		if self._out is None:
			self._out = np.concatenate(self._frames, axis=0)
			self._frames = None
		return self._out

	def __array__(self, dtype=None):
		out = self._force()
		if dtype is not None:
			out = out.astype(dtype)
		return out

	def __len__(self):
		if self._extremely_lazy:
			return len(self._frames)
		return len(self._force())

	def __getitem__(self, i):
		return self._force()[i]

	def count(self):
		if self.extremely_lazy:
			return len(self._frames)
		frames = self._force()
		return frames.shape[0]//3

	def frame(self, i):
		return self._force()[i*3:(i+1)*3]


def count_parameters(net, as_int=False):
	"""Returns number of params in network"""
	count = sum(p.numel() for p in net.parameters())
	if as_int:
		return count
	return f'{count:,}'


def save_obs(obs, fname='obs', resize_factor=None):
	assert obs.ndim == 3, 'expected observation of shape (C, H, W)'
	if isinstance(obs, torch.Tensor):
		obs = obs.detach().cpu()
	else:
		obs = torch.FloatTensor(obs)
	c,h,w = obs.shape
	if resize_factor is not None:
		obs = torchvision.transforms.functional.resize(obs, size=(h*resize_factor, w*resize_factor))
	if c == 3:
		torchvision.utils.save_image(obs/255., fname)
	elif c == 9:
		grid = torch.stack([obs[i*3:(i+1)*3] for i in range(3)], dim=0)
		grid = torchvision.utils.make_grid(grid, nrow=3)
		torchvision.utils.save_image(grid/255., fname)
	else:
		raise NotImplementedError('save_obs does not support other number of channels than 3 or 9')
