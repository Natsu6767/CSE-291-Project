from gym.envs.registration import register

REGISTERED_ROBOT_ENVS = False


def register_robot_envs(n_substeps=20, observation_type='image', reward_type='dense', image_size=84, use_xyz=False):
	global REGISTERED_ROBOT_ENVS
	if REGISTERED_ROBOT_ENVS:	
		return

	register(
		id='RobotLift-v0',
		entry_point='env.robot.lift:LiftEnv',
		kwargs=dict(
			xml_path='robot/lift.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotPickplace-v0',
		entry_point='env.robot.pick_place:PickPlaceEnv',
		kwargs=dict(
			xml_path='robot/pick_place.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotPeg-v0',
		entry_point='env.robot.peg_hole:PegEnv',
		kwargs=dict(
			xml_path='robot/peg_hole.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			
		)
	)

	register(
		id='RobotReach-v0',
		entry_point='env.robot.reach:ReachEnv',
		kwargs=dict(
			xml_path='robot/reach.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotReachmovingtarget-v0',
		entry_point='env.robot.reach:ReachMovingTargetEnv',
		kwargs=dict(
			xml_path='robot/reach.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotPush-v0',
		entry_point='env.robot.push:PushEnv',
		kwargs=dict(
			xml_path='robot/push.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	register(
		id='RobotPushnogoal-v0',
		entry_point='env.robot.push:PushNoGoalEnv',
		kwargs=dict(
			xml_path='robot/push.xml',
			n_substeps=n_substeps,
			observation_type=observation_type,
			reward_type=reward_type,
			image_size=image_size,
			use_xyz=use_xyz
		)
	)

	REGISTERED_ROBOT_ENVS = True
