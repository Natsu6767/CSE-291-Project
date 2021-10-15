from algorithms.sac import SAC
from algorithms.sacv2 import SACv2
from algorithms.drq import DrQ
from algorithms.svea import SVEA
from algorithms.multiview import MultiView
from algorithms.drq_multiview import DrQMultiView
from algorithms.drqv2 import DrQv2
from algorithms.sacv2_3d import SACv2_3D

algorithm = {
	'sac': SAC,
	'sacv2': SACv2,
	'sacv2_3d': SACv2_3D,
	'drq': DrQ,
	'svea': SVEA,
	'multiview': MultiView,
	'drq_multiview': DrQMultiView,
	'drqv2': DrQv2
}


def make_agent(obs_shape, in_shape, action_shape, args):
	return algorithm[args.algorithm](obs_shape, in_shape, action_shape, args)
