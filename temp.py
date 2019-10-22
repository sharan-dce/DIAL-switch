# from multi_step_mnist import MultiStepMNIST
# m = MultiStepMNIST('env-1', 1.0, -1.0)
# a = m.label
# for i in range(4):
# 	print(m.step(a))
# 	m.render()

# import time
# time.sleep(3)
import tensorflow as tf
from agent import Agent

agent = Agent(10, 10)

a = tf.zeros([1, 10])
b = tf.zeros([1, 12])
with tf.GradientTape() as tape:
	output = agent(a, b, tape)

grads = tape.gradient(output, agent.trainable_variables)
print(grads)