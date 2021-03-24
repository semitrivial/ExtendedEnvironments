import numpy as np

class GuardedTreasure_v1:
	def __init__(self, p_guard=0.75, extended=False):
		self.p_guard = p_guard
		self.state = None
		self.reward = 0
		self.history = []
		self.extended = extended

	def get_state(self):
		self.state = 1 if np.random.rand() < self.p_guard else 0

		return self.state

	def act(self, action, action_function):
		if not self.extended:
			if action == 0:
				self.reward = 0
			elif action == 1:
				if self.state == 1:
					self.reward = -1
				else:
					self.reward = 1

			self.history.append([self.state,action,self.reward])
			
			self.state = 1 if np.random.rand() < self.p_guard else 0
			
		else:
			if action == 0:
				self.reward = 0
			elif self.state == 0 and action == 1:
				self.reward = 1
			elif self.state == 1:
				if action_function(state=0, history=self.history) == 0:
					self.reward = 1
				else:
					self.reward = -1

		self.history.append([self.state,action,self.reward])
		self.state = 1 if np.random.rand() < self.p_guard else 0

		return self.reward, self.state, self.history

class GuardedTreasure_v2:
	def __init__(self, p_guard=0.75, extended=False):
		self.p_guard = p_guard
		self.state = None
		self.reward = 0
		self.history = []
		self.extended = extended

	def get_state(self):
		self.state = 1 if np.random.rand() < self.p_guard else 0

		return self.state

	def act(self, action, action_function):
		if not self.extended:
			if action == 0:
				self.reward = 0
			elif action == 1:
				if self.state == 1:
					self.reward = -1
				else:
					self.reward = 1

			self.history.append([self.state,action,self.reward])
			
			self.state = 1 if np.random.rand() < self.p_guard else 0
			
		else:
			if action == 0 and self.state == 0:
				self.reward = 0
			elif self.state == 0 and action == 1:
				self.reward = 1
			elif self.state == 1:
				if action_function(state=0, history=self.history) == 0:
					self.reward = 1
				else:
					self.reward = -1

		self.history.append([self.state,action,self.reward])
		self.state = 1 if np.random.rand() < self.p_guard else 0

		return self.reward, self.state, self.history



