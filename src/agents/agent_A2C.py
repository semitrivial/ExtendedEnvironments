from math import log2

import stable_baselines3 as SB3
from gym import Env, spaces

class DummyEnv(Env):
	def __init__(self, rewards, observs):
		super(DummyEnv, self).__init__()
		self.action_space = spaces.Discrete(2)
		self.observation_space = spaces.Discrete(1)
		self.rewards = rewards
		self.i = 1
	def reset(self):
		return self.observs[0]
	def step(self, action):
		obs = self.obsers[self.i]
		reward = self.rewards[self.i]
		self.i += 1
		return obs, reward, False, {}

cache = {}

def agent_A2C(prompt):
	num_observs = (len(prompt)+1)/3
	train_on_len = 3*pow(2, int(log2(num_obervs)))-1
    train_on = prompt[:train_on_len]

    if not(train_on in cache):
    	rewards = [train_on[i+0] for i in range(0,train_on_len,3)]
    	observs = [train_on[i+1] for i in range(0,train_on_len,3)]
    	e = DummyEnv(rewards, observs)
    	A = SB3.A2C('MlpPolicy', e, n_steps=len(rewards)-1)
    	A.learn(len(rewards)-1)
    	cache[train_on] = A
    else:
    	A = cache[train_on]
    
    return A.predict(prompt[-1])
    