import tensorflow as tf

class DoubleLinear:
	def __init__(self, units, activation = None):
		self.units = units
		self.w = tf.keras.layers.Dense(self.units)
		self.u = tf.keras.layers.Dense(self.units, use_bias = False)
		self.activation = activation

		self.__flag__ = True

	def __call__(self, input_w, input_u):
		pre_activation = self.w(input_w) + self.u(input_u)

		if self.__flag__:
			self.__flag__ = False
			self.trainable_variables = self.w.trainable_variables + self.u.trainable_variables

		return (self.activation(pre_activation) if self.activation != None else pre_activation)

class LSTM:
	def __init__(self, units):
		self.units = units

		self.forget_gate = DoubleLinear(units, tf.nn.sigmoid)
		self.output_gate = DoubleLinear(units, tf.nn.sigmoid)
		self.input_gate = DoubleLinear(units, tf.nn.sigmoid)
		self.cell_gate = DoubleLinear(units, tf.nn.tanh)

		self.__flag__ = True


	def __call__(self, input_tensor, tape = None):

		if self.__flag__:
			batch_size = input_tensor.shape[0]
			self.output = tf.zeros([batch_size, self.units])
			self.cell = tf.zeros([batch_size, self.units])
			if tape != None:
				tape.watch(self.cell)
				tape.watch(self.output)
		self.cell *= self.forget_gate(input_tensor, self.output)
		self.cell += self.input_gate(input_tensor, self.output) * self.cell_gate(input_tensor, self.output)
		self.output = self.output_gate(input_tensor, self.output) * tf.nn.tanh(self.cell)

		if self.__flag__:
			self.__flag__ = False
			self.trainable_variables = self.forget_gate.trainable_variables + self.output_gate.trainable_variables + self.input_gate.trainable_variables + self.cell_gate.trainable_variables

		return self.output

	def reset(self, batch_size = 1):
		self.__flag__ = True
