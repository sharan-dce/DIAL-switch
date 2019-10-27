import random
class Jail:
	def __init__(self, no_prisoners, limit):
		self.no_prisoners = no_prisoners
		self.active_prisoner = random.randint(0, no_prisoners - 1)		# the active one is in the room
		self.prisoner_mask = [int(i == self.active_prisoner) for i in range(self.no_prisoners)]
		self.actions_taken = 0
		self.limit = limit
		self.done = 0

	def step(self, action):		# action 0 or 1 : not declaring or declaring
								# returns (reward, done)
		assert self.done == 0, "Cannot be used further-- you are done!"
		assert action in [0, 1], "Actions can be 0 or 1... shape of action given: {}".format(action.shape)
		self.actions_taken += 1
		if action == 1:
			self.done = 1
			return ((1.0 if sum(self.prisoner_mask) == self.no_prisoners else -1.0), True)
		if self.actions_taken == self.limit:
			self.done = 1
			if sum(self.prisoner_mask) == self.no_prisoners:
				return (-0.1, True)
			else:
				return (0.0, True)
		self.active_prisoner = random.randint(0, self.no_prisoners - 1)
		self.prisoner_mask[self.active_prisoner] = 1
		return (0.0, False)

	def reset(self):
		self.done = 0
		self.active_prisoner = random.randint(0, self.no_prisoners - 1)		# the active one is in the room
		self.prisoner_mask = [int(i == self.active_prisoner) for i in range(self.no_prisoners)]
		self.actions_taken = 0
