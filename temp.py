# from multi_step_mnist import MultiStepMNIST
# m = MultiStepMNIST('env-1', 1.0, -1.0)
# a = m.label
# for i in range(4):
# 	print(m.step(a))
# 	m.render()

# import time
# time.sleep(3)
import tensorflow as tf
xz = tf.ones([])
c = tf.Variable(1.0)
with tf.GradientTape() as g:
  g.watch(xz)
  x = xz + c
  y = x * x
dy_dx, dy_dxz = g.gradient(y, [x, xz]) # Will compute to 6.0
print(dy_dx, dy_dxz)
