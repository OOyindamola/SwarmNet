import os
import argparse

import tensorflow as tf
from tensorflow import keras
import numpy as np
from os import path
import swarmnet

import configparser
import gym
import sys
import gym_swarm

from  velocity_controller import VelocityController


def pid_control(prev_state, next_state, kp, kd=0, ki=0):
	u = kp*(next_state-prev_state)

	return

def check_goal(current_state, next_state, min_dist_thresh):
	if next_state - current_state < min_dist_thresh:
		return True

	return False

def compute_edges(x, comm_radius2, mean_pooling= False):
	# Relative position among all pairs [q(j:2) - q(i:2)].
	n_agents = x.shape[0]
	xij = np.subtract(np.repeat(x[:,0], n_agents).reshape(n_agents, n_agents), x[:,0])
	yij = np.subtract(np.repeat(x[:,1], n_agents).reshape(n_agents, n_agents), x[:,1])

	# Relative distance among all pairs.
	dsqr = xij**2 + yij**2 #r2
	dist = np.sqrt(dsqr)

	np.fill_diagonal(dist, np.Inf)
	np.fill_diagonal(dsqr, np.Inf)

	adj_mat = (dsqr < comm_radius2).astype(float)

	adj_mat[1:n_agents,1:n_agents] = adj_mat[1:n_agents,1:n_agents] * 3 # influence from boids to boids
	adj_mat[:,0] = 0

	# Normalize the adjacency matrix by the number of neighbors - results in mean pooling, instead of sum pooling
	n_neighbors = np.reshape(np.sum(adj_mat, axis=1), (n_agents,1)) # correct - checked this
	n_neighbors[n_neighbors == 0] = 1
	adj_mat_mean = adj_mat / n_neighbors

	if mean_pooling:
		state_network = adj_mat_mean
	else:
		state_network = adj_mat
	return state_network


def process_state(env):
	goal= env.env.goal
	f = np.array([goal[0], goal[1], 0, 0])
	# print("xd:" ,env.env.xd)
	time_series =  np.vstack([f, env.env.xd])

	edges = compute_edges(time_series, env.comm_radius * env.comm_radius)

	edges = edges.reshape(1, 1,*edges.shape).astype(np.int)

	time_series = time_series.reshape(1, 1, *time_series.shape)


	# print(time_series.shape, edges.shape)

	return time_series, edges


if __name__ == '__main__':
	parser = argparse.ArgumentParser()

	parser.add_argument('--config', type=str,
						help='model config file')
	parser.add_argument('--segconfig', type=str,
						help='seg config file')
	parser.add_argument('--log-dir', type=str,
						help='log directory')

	parser.add_argument('--pred-steps', type=int, default=1,
						help='number of steps the estimator predicts for time series')

	parser.add_argument('--learning-rate', '--lr', type=float, default=None,
						help='learning rate')
	parser.add_argument('--dyn_edge', type= int, default=0,
						help='dyn_edge')
	parser.add_argument('--max-padding', type=int, default=None,
						help='max pad length to the number of agents dimension')

	ARGS = parser.parse_args()

	# ARGS.data_dir = os.path.expanduser(ARGS.data_dir)
	ARGS.config = os.path.expanduser(ARGS.config)
	ARGS.log_dir = os.path.expanduser(ARGS.log_dir)

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

	config_file = path.join(path.dirname(__file__), ARGS.segconfig)
	config = configparser.ConfigParser()
	config.read(config_file)
	args= config[config.sections()[0]]
	params = dict(config.items(config.sections()[0]))


	#creating environment and setting params
	env_name = params['env']
	gif_name = params['gif_name']
	centralized = args.getboolean('centralized')
	env = gym.make(env_name)
	env.env.params_from_cfg(args)
	nagents = int(params['n_agents']) +1
	ndims = 4



	###########################swarmnet
	model_params = swarmnet.utils.load_model_params(ARGS.config)
	model = swarmnet.SwarmNet.build_model(
		nagents, ndims, model_params, ARGS.pred_steps)

	swarmnet.utils.load_model(model, ARGS.log_dir)



	state, _ = env.reset()
	episode_reward = 0
	done = False
	steps = 0

	distThreshold, kp, kd, ki =  0.01, 0.5, 0.01, 0

	# main_state = state[0]
	while not done:

		data = process_state(env)

		_, edges = data


		input_data, expected_time_segs = swarmnet.data.preprocess_data(
			data, model_params['time_seg_len'], ARGS.pred_steps, edge_type=model_params['edge_type'],
			ground_truth=False,dyn_edge=ARGS.dyn_edge)

		prediction = model(input_data)

		pred_next_state = np.squeeze(prediction)
		# print(prediction.shape,prediction[0].shape )
		# print(env.env.goal)
		agent_nstate = pred_next_state[1:,:]
		# print(agent_nstate.shape)

		# vel_controller_ = VelocityController(env, agent_nstate, distThreshold,
	    #                 kp, kd, ki)

		# next_state, reward, done = vel_controller_.run()

		action = env.env.controller(True)
		next_state, reward, done, _ = env.step(action)


		env.env.set_xd(agent_nstate)

		# print("sja: ", env.env.xd)
		# print("dknln: ", agent_nstate)
		episode_reward += reward
		# state = env.x


		if steps % 10 == 0 and params['render'].lower() in ['true']:
			env.render('rgb_array')
			# print("rendering")

		if done or steps == params['max_path_length']:
			print(steps)
			break
		# if steps > 2:
		#     break

		steps = steps + 1
		print(steps)
	print("Episode Reward: ", episode_reward)

	print("final:", env.x[:,0:2])
	env.make_gif('./' + gif_name+'.gif')
	env.close()
