import cv2 as cv
from agent import Agent
import config
from mediator import Mediator
import numpy as np
import tensorflow as tf
from multi_step_mnist import MultiStepMNIST
import autoencoder
from dru import dru

def sample_action(alpha, q_values):	 # alpha is the probability with which the agent takes a random action
# returns shape [target_agent]
	result = []
	for L in range(0, len(q_values), 10):
		q_value = q_values[L: L + len(q_values)]

		if(np.random.uniform() < alpha):
			result.append(tf.constant(np.random.randint(q_value.shape[-1])))
		else:
			result.append(tf.argmax(q_value, axis = 1)[0])
	return result

def print_summary(episode, record):
	print('Train Episode {:7d}:	Reward: {:7.4f}'.format(episodes, sum(record['rewards'])))


def play_episode(agents, mediator, environments, encoder):
	assert len(agents) == len(environments), "Unequal no of environments and agents"
	for i in range(len(agents)):
		agents[i].reset()
		environments[i].reset()
	encodings = list(map(lambda env: encoder(env.image), environments))

	broadcasted_message = tf.constant(np.random.normal(scale = config.MESSAGE_NOISE_STDDEV, size = [1, config.AGENT_MESG_INPUT_DIMS]).astype(np.float32))

	cumulated_q_values = []		# time_steps x agents x (agents * 10)
	final_rewards = []			# environments x agents
	done = False
	cumulated_actions = []

	while not done:
		q_values = []
		messages = []
		for i, agent in enumerate(agents):
			q_value, message = agent(encodings[i], broadcasted_message)
			q_values.append(q_value)
			messages.append(message)
		cumulated_q_values.append(q_values)
		return
		# add noise to the channel
		noisy_messages = [dru(message, config.MESSAGE_NOISE_STDDEV, training) for message in messages]
		mediator_output = mediator(noisy_messages)
		noisy_mediator_output = dru(mediator_output, training)
		broadcasted_message = noisy_mediator_output
		sampled_actions = [sample_action(config.EXPLORE_PROB, q_value) for q_value in q_values]	# agents x agents(action)
		cumulated_actions.append(sampled_actions)		# time_steps x agents x agents(actions)
		for i in range(len(sampled_actions)):
			actions_on_environment_i = sampled_actions[:][i]

			rewards, done = environments[i].step(actions_on_environment_i)
			if rewards != None:
				final_rewards.append(rewards)

	# compute loss
	loss = tf.constant(0.0)
	for time_step in range(len(cumulated_actions) - 1):
		for agent in len(agents):
			target = cumulated_q_values[time_step + 1][agent]
			for L in range(0, len(10 * config.NO_AGENTS), 10):
				loss += tf.square(tf.reduce_max(target[L: L + 10]) - cumulated_q_values[time_step][agent][L: L + 10])
	return loss
			



def main():
	environments = []
	agents = []
	encoder = autoencoder.warmup()
	# Get environments
	for _ in range(config.NO_AGENTS):
		environments.append(MultiStepMNIST('Environment-{}'.format(_), config.CORRECT_REWARDS[_], config.INCORRECT_REWARDS[_]))
		agents.append(Agent(10 * (config.NO_AGENTS), config.AGENT_MESG_OUTPUT_DIMS))

	mediator = Mediator(config.AGENT_MESG_INPUT_DIMS)

	optimizer = tf.keras.optimizers.RMSprop(learning_rate = config.LEARNING_RATE)

	# # to initialize, we run a random episode
	play_episode(agents, mediator, environments, encoder)

	# episodes = 0
	# while(True):
	# 	with tf.GradientTape() as tape:
	# 		record = play_episode(agents, mediator, environments, encoder)
	# 		loss = sum(record['loss'])
	# 	trainable_variables = mediator.trainable_variables
	# 	for agent in agents:
	# 		trainable_variables += agent.trainable_variables

	# 	grads = tape.gradient(loss, trainable_variables)
	# 	optimizer.apply_gradients(zip(grads, trainable_variables))
	# 	episodes += 1
	# 	print_summary(episode, record)

	# 	if episodes % config.EPISODES_TO_TRAIN == 0:
	# 		record = play_episode(agents, mediator, pong)
	# 		print_summary(episode, record)

main()