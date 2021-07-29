import numpy as np
import os
from gym import utils
from env.robot.base_peg import BaseEnv, get_full_asset_path


class PegEnv(BaseEnv, utils.EzPickle):
	def __init__(self, xml_path, n_substeps=20, observation_type='image', reward_type='dense', image_size=84):
		BaseEnv.__init__(self,
			get_full_asset_path(xml_path),
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			reset_free=False,
			use_xyz=False
		)
		self.state_dim = (11,) if self.use_xyz else (8,)
		utils.EzPickle.__init__(self)

	def compute_reward(self, achieved_goal, goal, info):
		d = self.goal_distance(achieved_goal, goal, self.use_xyz)
		if self.reward_type == 'sparse':
			return -(d > self.distance_threshold).astype(np.float32)
		else:
			return np.around(-3*d - 0.5*np.square(self._pos_ctrl_magnitude), 4)

	def _get_achieved_goal(self):
		return self.sim.data.get_site_xpos('ee_2').copy()

	def _sample_goal(self):
		goal = self.center_of_table.copy()
		goal[0] += self.np_random.uniform(0.05,0.25, size=1)
		goal[1] += self.np_random.uniform(-0.25,0.25,size=1)
		#goal[0]+=0.05
		#goal[1]+=0.1
		return BaseEnv._sample_goal(self, goal)

	def _sample_initial_pos(self):
		gripper_target = self.center_of_table.copy()
		gripper_target[0] += self.np_random.uniform(0,0.05, size=1)
		gripper_target[1] += self.np_random.uniform(-0.1,0.1, size=1)
		gripper_target[2] += self.default_z_offset
		BaseEnv._sample_initial_pos(self, gripper_target)
