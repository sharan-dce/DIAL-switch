import tensorflow as tf
from lstm import LSTM
tf.keras.layers.LSTM = LSTM
class Agent:
	def __init__(self, output_q_values, output_mesg_dims):

		self.output_mesg_dims = output_mesg_dims
		self.output_q_values = output_q_values

		#Construct fully connected layers here and add trainable variables
		self.layers = []
		self.layers.append(tf.keras.layers.LSTM(64))
		self.layers.append(tf.keras.layers.LSTM(64))
		self.layers.append(tf.keras.layers.LSTM(32))

		self.q_values = tf.keras.layers.Dense(output_q_values)

		self.message = tf.keras.layers.Dense(output_mesg_dims, activation = tf.nn.tanh)

		self.all_layers = self.layers + [self.q_values, self.message]

		self.__flag__ = True

	def __call__(self, view_input, mesg_input, tape = None):	#returns q values, message_vector
		output_tensor = tf.concat([view_input, mesg_input], axis = -1)
		for layer in self.layers[: 3]:
			output_tensor = layer(output_tensor, tape)
		if self.__flag__:
			self.__flag__ = False
			self.trainable_variables = []
			for layer in self.all_layers:
				self.trainable_variables += layer.trainable_variables
		return self.q_values(output_tensor), self.message(output_tensor)

	def reset(self):
		for layer in self.all_layers[1:]:
			layer.reset()