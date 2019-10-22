import numpy as np
import cv2 as cv
class MultiStepMNIST:
	def __init__(self, window_name, correct_reward, incorrect_reward):
		self.window_name = window_name
		self.correct_reward = correct_reward
		self.incorrect_reward = incorrect_reward

		cv.namedWindow(self.window_name, cv.WINDOW_NORMAL)
		cv.resizeWindow(self.window_name, 100, 100)

		from tensorflow.keras.datasets import mnist
		self.images, self.labels = mnist.load_data()[0]
		self.reset()


	def reset(self):
		random_index = np.random.randint(len(self.images))
		self.image = np.asarray(self.images[random_index]) / 255.0
		self.label = self.labels[random_index]
		self.guesses = 0

	def step(self, actions):		# returns rewards, done
		self.guesses += 1
		if(self.guesses == 4):
			rewards = map(lambda x: self.correct_reward if action == x else self.incorrect_reward, actions)
			return rewards, True
		else:
			return None, False

	def render(self):
		cv.imshow(self.window_name, self.image)
		cv.waitKey(1)
