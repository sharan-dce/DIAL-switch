# from multi_step_mnist import MultiStepMNIST
# m = MultiStepMNIST('env-1', 1.0, -1.0)
# a = m.label
# for i in range(4):
# 	print(m.step(a))
# 	m.render()

# import time
# time.sleep(3)
import tensorflow as tf
import numpy as np
x = np.asarray(3.0)
with tf.GradientTape() as g:
  # g.watch(x)
  y = x * x
dy_dx = g.gradient(y, x) # Will compute to 6.0
print(dy_dx)