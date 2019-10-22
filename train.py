import cv2 as cv
from agent import Agent
import config
from mediator import Mediator
import numpy as np
import tensorflow as tf
from multi_step_mnist import MultiStepMNIST

def sample_action(alpha, q_values):	 # alpha is the probability with which the agent takes a random action
	if(np.random.uniform() < alpha):
		return np.random.randint(q_values.shape[-1])
	else:
		return tf.argmax(q_values, axis = 1).numpy().item()

def print_summary(record):
	print('Train Episode {:7d}:	Reward: {:7.4f}		Timesteps: {:4d}'.format(episodes, sum(record['rewards']), len(record['rewards'])))


def play_episode(agents, mediator, environment, tape = None):
	
	# these are q value tensor outputs and message tensor outputs at each timestep of an episode
	# we will record each tensor in these lists so as to access them later, or even return them
	record = {
		'q_values': [],
		'messages_a': [],
		'messages_b': [],
		'actions_a': [],
		'actions_b': [],
		'rewards': [],
		'message_broadcasts': [tf.zeros([1, config.AGENT_MESG_INPUT_DIMS])]
	}
	done = False
	if tape != None:
		tape.watch(record['message_broadcasts'][-1])
	time = 0
	while not done:
		if tape != None:
			tape.watch(states[-1])
		state_a = states[-1][:, : config.VIEW_L, :]
		state_b = states[-1][:, -config.VIEW_R: , :]
		q_values, message = agents[0].forward_prop(state_a, record['message_broadcasts'][-1])
		record['q_values_a'].append(q_values)
		record['messages_a'].append(message)
		q_values, message = agents[1].forward_prop(state_b, record['message_broadcasts'][-1])
		record['q_values_b'].append(q_values)
		record['messages_b'].append(message)
		# now run through mediator
		message_list = [record['messages_a'][-1], record['messages_b'][-1]]
		message = mediator.forward_prop(message_list)
		record['message_broadcasts'].append(message)

		# now get the next state, run this info through the agents and remove the preivous to previous screen from states vairable
		action_a = sample_action(config.EXPLORE_PROB, record['q_values_a'][-1])
		action_b = sample_action(config.EXPLORE_PROB, record['q_values_b'][-1])
		record['actions_a'].append(action_a)
		record['actions_b'].append(action_b)
		pong.render()
		reward, done = pong.step(action_a, action_b)
		record['rewards'].append(tf.Variable(reward))
		new_state = np.expand_dims(np.expand_dims(pong.get_screen(), axis = -1), axis = 0)
		states.append(tf.concat([states[-1][:, :, :, -1:], new_state], axis = -1))
		time += 1

	# compute losses
	if get_loss:
		if tape != None:
			tape.watch(record['rewards'][-1])
		loss_a = tf.square(record['q_values_a'][-1][0, record['actions_a'][-1]] - record['rewards'][-1])
		loss_b = tf.square(record['q_values_b'][-1][0, record['actions_b'][-1]] - record['rewards'][-1])
		for t in range(len(record['q_values_a']) - 1):
			if tape != None:
				tape.watch(record['rewards'][t])
			loss_a += tf.square(record['q_values_a'][t][0, record['actions_a'][t]] - (record['rewards'][t] + config.DISCOUNT_FACTOR * tf.reduce_max((record['q_values_a'][t + 1])) )	)
			loss_b += tf.square(record['q_values_b'][t][0, record['actions_b'][t]] - (record['rewards'][t] + config.DISCOUNT_FACTOR * tf.reduce_max((record['q_values_b'][t + 1])) )	)
		record['loss_a'] = loss_a
		record['loss_b'] = loss_b
	return record


def main():
	environments = []
	agents = []
	# Get environments
	for _ in range(NO_AGENTS):
		environments.append(MultiStepMNIST('Environment-{}'.format(_), config.CORRECT_REWARDS[_], config.INCORRECT_REWARDS[_]))
		agents.append(Agent(10 * (config.NO_AGENTS), config.AGENT_MESG_OUTPUT_DIMS))

	mediator = Mediator(config.AGENT_MESG_INPUT_DIMS)

	optimizer = tf.keras.optimizers.RMSprop(learning_rate = config.LEARNING_RATE)

	# to initialize, we run a random episode
	play_episode(agents, mediator, environments)

	episodes = 0
	while(True):
		with tf.GradientTape() as tape:
			record = play_episode(agents, mediator, environments, tape)
			loss = sum(record['loss'])
		trainable_variables = []
		for agent in agents:
			trainable_variables += list(map(lambda x: x.trainable_variables, agent.all_layers))
		trainable_variables += list(map(lambda x: x.trainable_variables, mediator.all_layers))
		grads = tape.gradient(loss, trainable_variables)
		optimizer.apply_gradients(zip(grads, trainable_variables))
		episodes += 1
		print_summary(record)

		if episodes % config.EPISODES_TO_TRAIN == 0:
			record = play_episode(agents, mediator, pong, False)
			print_summary(record)

main()