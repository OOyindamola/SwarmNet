import os
import argparse

import tensorflow as tf
from tensorflow import keras
import numpy as np

import swarmnet


def eval_baseline(eval_data):
	time_segs = eval_data[0]
	return np.mean(np.square(time_segs[:, :-1, :, :] -
							 time_segs[:, 1:, :, :]))

def eval_baseline_(eval_data, prediction):
	time_segs = eval_data[0][0]
	prediction = np.squeeze(prediction[:2501,:,:,:])
	print(time_segs.shape, prediction.shape)
	return np.mean(np.square(time_segs[:, :, :] -
							 prediction[:, :, :]))

def main():
	if ARGS.train:
		prefix = 'train'
	elif ARGS.eval:
		prefix = 'valid'
	else:
		prefix = 'test'

	model_params = swarmnet.utils.load_model_params(ARGS.config)

	# print("model_p: ", model_params)
	if ARGS.learning_rate is not None:
		model_params['learning_rate'] = ARGS.learning_rate

	# data contains edge_types if `edge=True`.
	print("ARGS.dyn_edge: ", ARGS.dyn_edge)
	data = swarmnet.data.load_data(ARGS.data_dir,
								   prefix=prefix, size=ARGS.data_size, padding=ARGS.max_padding,dyn_edge=ARGS.dyn_edge)

	max_pred = 2
	if ARGS.test:
		max_pred = 2

	for i in range(1, max_pred, 2):
		if ARGS.train or ARGS.eval:
			ARGS.pred_steps = i
			epochs = int(ARGS.epochs/ARGS.pred_steps)

		print("pred_steps: ", ARGS.pred_steps)

		# input_data: a list which is [time_segs, edge_types] if `edge_type` > 1, else [time_segs]
		input_data, expected_time_segs = swarmnet.data.preprocess_data(
			data, model_params['time_seg_len'], ARGS.pred_steps, edge_type=model_params['edge_type'], ground_truth=not ARGS.test,dyn_edge=ARGS.dyn_edge)
		print(f"\n{prefix.capitalize()} data from {ARGS.data_dir} processed.\n")

		nagents, ndims = data[0].shape[-2:]
		#
		model = swarmnet.SwarmNet.build_model(
			nagents, ndims, model_params, ARGS.pred_steps)
		model.summary()


		if ARGS.train:
			swarmnet.utils.load_model(model, ARGS.log_dir)
			checkpoint = swarmnet.utils.save_model(model, ARGS.log_dir)

			# Freeze some of the layers according to train mode.
			if ARGS.train_mode == 1:
				model.conv1d.trainable = True

				model.graph_conv.edge_encoder.trainable = True
				model.graph_conv.node_decoder.trainable = False

			elif ARGS.train_mode == 2:
				model.conv1d.trainable = False

				model.graph_conv.edge_encoder.trainable = False
				model.graph_conv.node_decoder.trainable = True

			model.fit(input_data, expected_time_segs,
					  epochs=epochs, batch_size=ARGS.batch_size,
					  callbacks=[checkpoint])

			checkpoint_ = swarmnet.utils.save_model_(model, ARGS.log_dir, str(ARGS.pred_steps))

			#valid
			# if i % 5:


			# prefix ='valid'
			# print("ARGS.dyn_edge: ", ARGS.dyn_edge)
			# data_ = swarmnet.data.load_data(ARGS.data_dir,
			#                                prefix=prefix, size=ARGS.data_size, padding=ARGS.max_padding,dyn_edge=ARGS.dyn_edge)
			#
			# input_data_, expected_time_segs_ = swarmnet.data.preprocess_data(
			#     data_, model_params['time_seg_len'], ARGS.pred_steps, edge_type=model_params['edge_type'], ground_truth=not ARGS.test,dyn_edge=ARGS.dyn_edge)
			# print(f"\n{prefix.capitalize()} data from {ARGS.data_dir} processed.\n")
			#
			#
			# result = model.evaluate(
			#     input_data, expected_time_segs, batch_size=ARGS.batch_size)
			# # result = MSE
			# baseline = eval_baseline(data)
			# print('Baseline:', baseline, '\t| MSE / Baseline:', result / baseline)
			#
			# del data_, result, input_data_, expected_time_segs_

		elif ARGS.eval:
			swarmnet.utils.load_model(model, ARGS.log_dir,"9")
			result = model.evaluate(
				input_data, expected_time_segs, batch_size=ARGS.batch_size)
			prediction = model.predict(input_data)

			# result = MSE

			baseline = eval_baseline(data)
			# error = eval_baseline_(data, prediction)
			print('Baseline:', baseline , '\t| MSE:', result, '\t| MSE / Baseline:', result / baseline)
			# print('Error:', error)

		elif ARGS.test:
			swarmnet.utils.load_model(model, ARGS.log_dir)
			print("input_data: ", input_data[0].shape)
			prediction = model.predict(input_data)
			# print("bbb")
			np.save(os.path.join(ARGS.log_dir,
					f'prediction_{ARGS.pred_steps}.npy'), prediction)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data-dir', type=str,
						help='data directory')
	parser.add_argument('--data-size', type=int, default=None,
						help='optional data size cap to use for training')
	parser.add_argument('--config', type=str,
						help='model config file')
	parser.add_argument('--log-dir', type=str,
						help='log directory')
	parser.add_argument('--epochs', type=int, default=1,
						help='number of training steps')
	parser.add_argument('--pred-steps', type=int, default=1,
						help='number of steps the estimator predicts for time series')
	parser.add_argument('--batch-size', type=int, default=128,
						help='batch size')
	parser.add_argument('--learning-rate', '--lr', type=float, default=None,
						help='learning rate')
	parser.add_argument('--train', action='store_true', default=False,
						help='turn on training')
	parser.add_argument('--dyn_edge', type= int, default=0,
						help='dyn_edge')
	parser.add_argument('--train-mode', type=int, default=0,
						help='train mode determines which layers are frozen: '
							 '0 - all layers are trainable; '
							 '1 - conv1d layers and edge encoders are trainable; '
							 '2 - edge encoders and node encoder are trainable.')
	parser.add_argument('--max-padding', type=int, default=None,
						help='max pad length to the number of agents dimension')
	parser.add_argument('--eval', action='store_true', default=False,
						help='turn on evaluation')
	parser.add_argument('--test', action='store_true', default=False,
						help='turn on test')
	ARGS = parser.parse_args()

	ARGS.data_dir = os.path.expanduser(ARGS.data_dir)
	ARGS.config = os.path.expanduser(ARGS.config)
	ARGS.log_dir = os.path.expanduser(ARGS.log_dir)

	os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
	os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'

	main()
