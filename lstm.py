import tensorflow as tf

class DoubleLinear:
	def __init__(self, units, activation = None):
		self.units = units
		self.w = tf.keras.layers.Dense(self.units, kernel_initializer = tf.random_normal_initializer)
		self.u = tf.keras.layers.Dense(self.units, use_bias = False, kernel_initializer = tf.random_normal_initializer)
		self.activation = activation

		self.__flag__ = True

	def __call__(self, input_w, input_u):
		pre_activation = self.w(input_w) + self.u(input_u)

		if self.__flag__:
			self.__flag__ = False
			self.trainable_variables = self.w.trainable_variables + self.u.trainable_variables

		return (self.activation(pre_activation) if self.activation != None else pre_activation)

	def get_weights(self):
		return [self.w.get_weights(), self.u.get_weights()]

	def set_weights(self, weights):
		self.w.set_weights(weights[0])
		self.u.set_weights(weights[1])

class LSTM:
	def __init__(self, units):
		self.units = units

		self.forget_gate = DoubleLinear(units, tf.nn.sigmoid)
		self.output_gate = DoubleLinear(units, tf.nn.sigmoid)
		self.input_gate = DoubleLinear(units, tf.nn.sigmoid)
		self.cell_gate = DoubleLinear(units, tf.nn.tanh)

		self.__flag__ = True


	def __call__(self, input_tensor):

		if self.__flag__:
			batch_size = input_tensor.shape[0]
			self.output = tf.random.normal([batch_size, self.units])
			self.cell = tf.random.normal([batch_size, self.units])
		self.cell *= self.forget_gate(input_tensor, self.output)
		self.cell += self.input_gate(input_tensor, self.output) * self.cell_gate(input_tensor, self.output)
		self.output = self.output_gate(input_tensor, self.output) * tf.nn.tanh(self.cell)

		if self.__flag__:
			self.__flag__ = False
			self.trainable_variables = self.forget_gate.trainable_variables + self.output_gate.trainable_variables + self.input_gate.trainable_variables + self.cell_gate.trainable_variables

		return self.output

	def reset(self, batch_size = 1):
		self.__flag__ = True

	def get_weights(self):
		weights = []
		weights.append(self.forget_gate.get_weights())
		weights.append(self.output_gate.get_weights())
		weights.append(self.input_gate.get_weights())
		weights.append(self.cell_gate.get_weights())
		return weights

	def set_weights(self, weights):
		self.forget_gate.set_weights(weights[0])
		self.output_gate.set_weights(weights[1])
		self.input_gate.set_weights(weights[2])
		self.cell_gate.set_weights(weights[3])

class GRU:
	def __init__(self, units):
		self.units = units

		self.z = DoubleLinear(units, tf.nn.sigmoid)
		self.r = DoubleLinear(units, tf.nn.sigmoid)
		self.cell_gate = DoubleLinear(units, tf.nn.tanh)
		self.__flag__ = True


	def __call__(self, input_tensor):

		if self.__flag__:
			batch_size = input_tensor.shape[0]
			self.output = tf.random.normal([batch_size, self.units])

		zt = self.z(input_tensor, self.output)
		rt = self.r(input_tensor, self.output)
		self.output = (1.0 - zt) * self.output + zt * self.cell_gate(input_tensor, self.output * rt)

		if self.__flag__:
			self.__flag__ = False
			self.trainable_variables = self.z.trainable_variables + self.r.trainable_variables + self.cell_gate.trainable_variables

		return self.output

	def reset(self, batch_size = 1):
		self.__flag__ = True

	def get_weights(self):
		weights = []
		weights.append(self.z.get_weights())
		weights.append(self.r.get_weights())
		weights.append(self.cell_gate.get_weights())
		return weights

	def set_weights(self, weights):
		self.z.set_weights(weights[0])
		self.r.set_weights(weights[1])
		self.cell_gate.set_weights(weights[2])