import numpy as np
import os
from gym import utils
from env.robot.base import BaseEnv, get_full_asset_path



class ReachWallEnv(BaseEnv, utils.EzPickle):
	def __init__(self, xml_path, cameras, n_substeps=20, observation_type='image', reward_type='dense', image_size=84, use_xyz=False, render=False):
		BaseEnv.__init__(self,
			get_full_asset_path(xml_path),
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			reset_free=False,
			cameras=cameras,
			render=render,
			use_xyz=use_xyz
		)
		self.state_dim = (11,) if self.use_xyz else (8,)
		utils.EzPickle.__init__(self)

	def compute_reward(self, achieved_goal, goal, info):
		d = self.goal_distance(achieved_goal, goal, self.use_xyz)
		if self.reward_type == 'sparse':
			return -(d > self.distance_threshold).astype(np.float32)
		else:
			return np.around(-3*d - 0.5*np.square(self._pos_ctrl_magnitude), 4)

	def _get_state_obs(self):
		cot_pos = self.center_of_table.copy()
		dt = self.sim.nsubsteps * self.sim.model.opt.timestep

		eef_pos = self.sim.data.get_site_xpos('ee_2') - cot_pos
		eef_velp = self.sim.data.get_site_xvelp('ee_2') * dt
		goal_pos = self.goal - cot_pos
		gripper_angle = self.sim.data.get_joint_qpos('right_outer_knuckle_joint')

		if not self.use_xyz:
			eef_pos = eef_pos[:2]
			eef_velp = eef_velp[:2]
			goal_pos = goal_pos[:2]

		values = np.array([
			self.goal_distance(eef_pos, goal_pos, self.use_xyz),
			gripper_angle
		])

		return np.concatenate([
			eef_pos, eef_velp, goal_pos, values
		], axis=0)

	def _get_achieved_goal(self):
		return self.sim.data.get_site_xpos('ee_2').copy()

	def _sample_goal(self):
		goal = self.center_of_table.copy() + np.array([0.3, 0, 0])
		goal[0] += self.np_random.uniform(-0.05,-0.2 ,size=1)
		goal[1] += self.np_random.uniform(-0.2, 0.2, size=1)
		goal[2] += 0.05
		return BaseEnv._sample_goal(self, goal)

	def _sample_initial_pos(self):
		gripper_target = self.center_of_table.copy() - np.array([0.3, 0, 0])
		gripper_target[0] += self.np_random.uniform(-0.15, -0.05, size=1)
		gripper_target[1] += self.np_random.uniform(-0.05, 0.05, size=1)
		gripper_target[2] += self.default_z_offset
		if self.use_xyz:
			gripper_target[2] += self.np_random.uniform(0, 0.1, size=1)
		BaseEnv._sample_initial_pos(self, gripper_target)


class ReachMovingTargetEnv(ReachEnv):
	def __init__(self, *args, **kwargs):
		super().__init__(*args, **kwargs)
		self.set_velocity()
		
	def set_velocity(self):
		self.curr_vel = 0.0025 * np.ones(2)

	def _sample_goal(self):
		self.set_velocity()
		return ReachEnv._sample_goal(self)

	def _step_callback(self):
		self.set_goal()

	def set_goal(self):
		curr_goal = self.goal
		
		if (curr_goal[0] >= 1.4 and self.curr_vel[0] > 0) \
				or curr_goal[0] <= 1.2 and self.curr_vel[0] < 0:
			self.curr_vel[0] = -1 * self.curr_vel[0]
		if (curr_goal[1] >= 0.2 and self.curr_vel[1] > 0) \
				or curr_goal[1] <= -0.2 and self.curr_vel[1] < 0:
			self.curr_vel[1] = -1 * self.curr_vel[1]
		self.goal[0] += self.curr_vel[0]
		self.goal[1] += self.curr_vel[1]
