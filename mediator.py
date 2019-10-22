import tensorflow as tf

class Mediator:
	def __init__(self, output_dims):
		self.output_dims = output_dims
		self.layers = []
		self.layers.append(tf.keras.layers.Dense(units = 8, activation = tf.nn.relu))
		self.layers.append(tf.keras.layers.Dense(units = 16, activation = tf.nn.relu))
		self.layers.append(tf.keras.layers.Dense(units = 8, activation = tf.nn.relu))
		self.layers.append(tf.keras.layers.Dense(units = output_dims, activation = tf.nn.tanh))

		self.__flag__ = True

	def __call__(self, agent_inputs):
		output_tensor = tf.concat(agent_inputs, axis = 1)
		for layer in self.layers:
			output_tensor = layer(output_tensor)

		if self.__flag__:
			self.__flag__ = False
			self.trainable_variables = []
			for layer in self.layers:
				self.trainable_variables += layer.trainable_variables

		return output_tensor
