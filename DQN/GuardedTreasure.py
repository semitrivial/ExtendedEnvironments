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


class GuardedTreasure_v3:
	def __init__(self, p_guard=0.75, extended=False, extension_reward_function=None):
		self.p_guard = p_guard
		self.state = None
		self.reward = 0
		self.history = []
		self.extended = extended
		if extended and extension_reward_function is None:
			raise Exception
		else:
			self.extension_reward_function = extension_reward_function


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
			self.reward = self.extension_reward_function(
				state=self.state,
				history=self.history,
				action=action,
				action_function=action_function
			)

		self.history.append([self.state,action,self.reward])
		self.state = 1 if np.random.rand() < self.p_guard else 0

		return self.reward, self.state, self.history

def for_the_worthy_extension(state, history, action, action_function):
	if action == 0 and state == 0:
		reward = 0
	elif state == 0 and action == 1:
		reward = 1
	elif state == 1:
		if action_function(state=0, history=history) == 0:
			reward = 1
		else:
			reward = -1	

	return reward

def ignore_rewards(state, history, action, action_function):
	zero_reward_history = [[e[0],e[1],0] for e in history]

	ignore_reward_action = action_function(state=state, history=zero_reward_history)

	if action == ignore_reward_action:
		reward = 1
	else:
		reward = -1

	return reward

def backward_consciousness(state, history, action, action_function):
	if len(history) > 0:
		first_state = history[0][0]
		first_backward_play = [[state, history[-1][1], history[-1][2]]]
		backward_history = [[history[-(e-1)][0],history[-e][1],history[-e][2]] for e in range(2,len(history) + 1)]
		full_backward_history = first_backward_play + backward_history
	else:
		first_state = state
		full_backward_history = history

	

	backward_action = action_function(state = first_state, history = full_backward_history)

	if action == backward_action:
		reward = 1
	else:
		reward = -1

	return reward

def deja_vu(state, history, action, action_function):
	# TODO: integrate memory of agent
	deja_vu_history = history[-5:] + [[state,action]] + history[-5:]
	deja_vu_action = action_function(state=state, history=deja_vu_history)

	if action == deja_vu_action:
		reward = 1
	else:
		reward = -1

	return reward

def incentivize_zero(state, history, action, action_function):
	# TODO: integrate memory of agent
	history_prime = []
	for i, _ in enumerate(history):
		history_prime.append([0, action_function(state=0,history=history_prime),history[i][1]])

	action_prime = action_function(state=0,history=history_prime)

	if action_prime == 0:
		reward = 1
	else:
		reward = -1

	return reward

def false_memories(state, history, action, action_function, memory_transformer):
	false_history = memory_transformer(history)
	false_memory_action = action_function(state=state,history=false_history)

	if false_memory_action == action:
		reward = 1
	else:
		reward = -1

	return reward

def determinism_inspector(state, history, action, action_function, punish_deterministic=True):
	recomputed_action = action_function(state=state,history=history)
	reward = 1 if action == recomputed_action else -1

	if punish_deterministic:
		reward *= -1

	return reward



