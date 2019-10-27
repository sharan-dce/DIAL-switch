from lstm import GRU
import numpy as np
import random
import tensorflow as tf
layers = tf.keras.layers
from environment import Jail
import logging
tf.get_logger().setLevel(logging.ERROR)

NO_AGENTS = 4
TEST_RUNS = 100
TRAIN_RUNS = 100
LEARNING_RATE = 0.0005
UPDATE_TARGET = 100
DUMMY_MESSAGE = tf.constant([[0.0]])

in_room = tf.constant([[1.0]])
outside_room = tf.constant([[0.0]])

def dru(message, stddev):
	noise = tf.random.normal(message.shape, mean = 0.0, stddev = stddev)
	return noise + message

def sample_action(q_values, alpha):
	random_sample = np.random.uniform(0, 1.0)
	if random_sample < alpha:
		return tf.random.uniform([], 0, 2, dtype = tf.int64)
	return tf.argmax(q_values, axis = 1)[0]

class Agent:
	def __init__(self):
		self.layers = []
		self.layers.append(GRU(64))
		self.layers.append(GRU(128))
		self.lstm_cnt = len(self.layers)
		self.layers.append(layers.Dense(32, kernel_initializer = tf.random_normal_initializer, activation = tf.nn.relu))
		self.layers.append(layers.Dense(16, kernel_initializer = tf.random_normal_initializer, activation = tf.nn.relu))
		self.q_values = layers.Dense(2, kernel_initializer = tf.random_normal_initializer, bias_initializer = tf.random_normal_initializer)
		self.message = layers.Dense(1, activation = tf.nn.sigmoid, kernel_initializer = tf.random_normal_initializer, bias_initializer = tf.random_normal_initializer)

		self.all_layers = self.layers + [self.q_values, self.message]

	def __call__(self, room, message):		# q_values, message
		input_tensor = tf.concat([room, message], axis = 1)
		for layer in self.layers:
			input_tensor = layer(input_tensor)
		return self.q_values(input_tensor), self.message(input_tensor)

	def trainable_variables(self):
		trainable_vars = []
		for layer in self.all_layers:
			trainable_vars += layer.trainable_variables
		return trainable_vars

	def reset(self):
		for lstm_layer in self.layers[: self.lstm_cnt]:
			lstm_layer.reset()

	def get_weights(self):
		weights = []
		for layer in self.all_layers:
			weights.append(layer.get_weights())
		return weights

	def set_weights(self, weights):
		for i, weight in enumerate(weights):
			self.all_layers[i].set_weights(weight)


def initializing_run(agents):
	message = tf.random.normal([1, 1])
	room = tf.zeros([1, 1])
	for agent in agents:
		agent(room, message)

def play_episode(environment, agents, target_agents, training):
	environment.reset()
	for agent in agents + target_agents:
		agent.reset()
	done = False
	message = tf.random.uniform([1, 1])
	target_message = tf.random.normal([1, 1])
	q_values_log = []
	target_q_values_log = []
	rewards = []
	actions = []
	mask = [0 for _ in range(NO_AGENTS)]
	loss = tf.constant(0.0)
	while not done:
		q_values, message = agents[environment.active_prisoner](in_room, message)
		target_q_values, target_message = target_agents[environment.active_prisoner](in_room, target_message)
		for i in range(len(agents)):
			if i != environment.active_prisoner:
				agents[i](outside_room, DUMMY_MESSAGE)
				target_agents[i](outside_room, DUMMY_MESSAGE)
		mask[environment.active_prisoner] = 1
		if training:
			message = dru(message, 1.0)
			action = sample_action(q_values, 0.05)
		else:
			message = tf.constant([[0.]]) if message.numpy().any() < 0.5 else tf.constant([[1.]])
			action = tf.argmax(q_values, axis = 1)[0]
		actions.append(action)
		reward, done = environment.step(action.numpy().item())
		q_values_log.append(q_values)
		target_q_values_log.append(target_q_values)
		rewards.append(reward)

	loss = tf.constant(0.0)
	for i in range(len(rewards)):
		if i + 1 == len(rewards):
			loss += tf.square(q_values_log[i][0][actions[i]] - rewards[i])
		else:
			loss += tf.square(q_values_log[i][0][actions[i]] - rewards[i] - tf.reduce_max(target_q_values_log[i + 1]))
	return loss, len(actions), mask, rewards
def desc(prisoner, discretised_message, final_bulb_state, declared):
	print('{:3d} --> Bulb state: {:3.3f} --> Discretised Mesg: {:3.1f} --> Declared: {:3d}'.format(prisoner, discretised_message, final_bulb_state, declared))

def visual(environment, agents):
	environment.reset()
	for agent in agents:
		agent.reset()
	done = False
	message = tf.random.uniform([1, 1])
	actions = []
	mask = [0 for _ in range(NO_AGENTS)]
	while done == False:
		q_values, message = agents[environment.active_prisoner](in_room, message)
		for i in range(len(agents)):
			if i != environment.active_prisoner:
				agents[i](outside_room, DUMMY_MESSAGE)
		mask[environment.active_prisoner] = 1
		cmes = message
		message = tf.constant([[0.]]) if message.numpy().item() < 0.5 else tf.constant([[1.]])
		action = tf.argmax(q_values, axis = 1)[0]
		desc(environment.active_prisoner, cmes.numpy().item(), message.numpy().item(), action.numpy().item())
		reward, done = environment.step(action.numpy())

def main():
	save_threshold = 90
	jail = Jail(NO_AGENTS, 4 * NO_AGENTS)
	test_env = Jail(NO_AGENTS, 30)
	agents = [Agent() for _ in range(NO_AGENTS)]
	target_agents = [Agent() for _ in range(NO_AGENTS)]
	for i in range(NO_AGENTS):
		target_agents[i].set_weights(agents[i].get_weights())
	initializing_run(agents)
	optimizer = tf.keras.optimizers.RMSprop(LEARNING_RATE)
	episodes = 0

	### initial performance

	mean_rewards = 0.0
	succ = 0
	for _ in range(TEST_RUNS):
		print('\r({} Agents) 	Episode {}	 	Test {}'.format(NO_AGENTS, episodes, _), end = '')
		loss, time_steps, mask, rewards = play_episode(test_env, agents, target_agents, training = False)
		mean_rewards += sum(rewards)
		if sum(rewards) == 1.0:
			succ += 1
	print('\r({} Agents) ({} episodes):		Mean reward over {:4d} tests: {:5.2f} 		Successful attempts: {:5d}		Learning-Rate: {:5.5f}'.format(NO_AGENTS, episodes, TEST_RUNS, mean_rewards / TEST_RUNS, succ, LEARNING_RATE))
	visual(jail, agents)

	###
	while True:
		episodes += 1
		with tf.GradientTape(0) as tape:
			loss, time_steps, mask, rewards = play_episode(jail, agents, target_agents, training = True)
		trainable_variables = []
		for i, j in enumerate(mask):
			if j == 1:
				trainable_variables += agents[i].trainable_variables()
		gradients = tape.gradient(loss, trainable_variables)
		optimizer.apply_gradients(zip(gradients, trainable_variables))
		print('\r({} Agents) Episode {}'.format(NO_AGENTS, episodes), end = '')
		if episodes % TRAIN_RUNS == 0:
			mean_rewards = 0.0
			succ = 0
			for _ in range(TEST_RUNS):
				print('\r({} Agents) 	Episode {}	 	Test {}'.format(NO_AGENTS, episodes, _), end = '')
				loss, time_steps, mask, rewards = play_episode(test_env, agents, target_agents, training = False)
				mean_rewards += sum(rewards)
				if sum(rewards) == 1.0:
					succ += 1
			print('\r({} Agents) ({} episodes):		Mean reward over {:4d} tests: {:5.2f} 		Successful attempts: {:5d}		Learning-Rate: {:5.5f}'.format(NO_AGENTS, episodes, TEST_RUNS, mean_rewards / TEST_RUNS, succ, LEARNING_RATE))
			visual(jail, agents)

		if episodes % UPDATE_TARGET == 0:
			for i in range(NO_AGENTS):
				target_agents[i].set_weights(agents[i].get_weights())


if __name__ == '__main__':
	main()