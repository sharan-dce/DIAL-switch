import tensorflow as tf
import numpy as np

def dru(message, stddev, tape = None):
	noise = tf.random.normal(message.shape, mean = 0.0, stddev = stddev)
	if tape != None:
		tape.watch(noise)
	return noise + message