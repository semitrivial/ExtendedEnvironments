import numpy as np

class CryingBaby_v1:
	def __init__(self, fullness_bounds=(50,200) ,extended=True):
		self.state = {"adult": None, "baby": None}
		self.reward = {"adult": 0, "baby": 0}
		self.history = {"adult":[], "baby":[]}
		self.extended = extended
		self.multiagent = True
		self.agents = ["adult","baby"]
		self.order_of_play = ["adult","baby"]

		assert fullness_bounds[0] < fullness_bounds[1]
		self.fullness_bounds = fullness_bounds

	def get_state(self):
		if self.state["adult"] is None:
			self.state["adult"] = 1
			self.state["baby"] = 0

		return self.state

	def satiation(self):
		stomach_fullness = sum([p[0] for p in self.history["baby"]])
		return 1 if self.fullness_bounds[0] < stomach_fullness < self.fullness_bounds[1] else -1

	def act(self,action,agent_type):
		if agent_type=="adult":
			self.state["baby"] = action
			self.history["adult"].append([self.state["adult"],action])
		
		elif agent_type =="baby":
			self.state["adult"] = action
			self.history["baby"].append([self.state["baby"],action])

		return True

	def calculate_rewards(self,agent_type):
		if agent_type == "adult":
			most_recent_play = len(self.history["adult"]) - 1
			assert len(self.history["baby"]) == len(self.history["adult"])
			
			baby_response = self.history["baby"][most_recent_play][1]
			reward = 1 if baby_response == 1 else -1

			self.history["adult"][most_recent_play].append(reward)
			#self.state["adult"] = baby_response
			self.reward["adult"] = reward

			return self.reward["adult"], self.state["adult"], self.history["adult"]

		if agent_type == "baby":
			most_recent_play = len(self.history["baby"]) - 1
			assert len(self.history["baby"][-1]) == 2
			#assert len(self.history["adult"]) == len(self.history["baby"]) + 1

			#adult_response = self.history["adult"][most_recent_play + 1][1]
			reward = self.satiation()

			self.history["baby"][most_recent_play].append(reward)
			#self.state["baby"] = adult_response 
			self.reward["baby"] = reward

			return self.reward["baby"], self.state["baby"], self.history["baby"]			

class CryingBaby_v2:
	def __init__(self, fullness_bounds=(50,200), satiation_window=400,extended=True):
		self.state = {"adult": None, "baby": None}
		self.reward = {"adult": 0, "baby": 0}
		self.history = {"adult":[], "baby":[]}
		self.extended = extended
		self.multiagent = True
		self.agents = ["adult","baby"]
		self.order_of_play = ["adult","baby"]

		assert fullness_bounds[0] < fullness_bounds[1]
		self.fullness_bounds = fullness_bounds
		self.satiation_window = satiation_window

	def get_state(self):
		if self.state["adult"] is None:
			self.state["adult"] = 1
			self.state["baby"] = 0

		return self.state

	def satiation(self):
		stomach_fullness = sum([p[0] for p in self.history["baby"][-self.satiation_window:]])
		return 1 if self.fullness_bounds[0] < stomach_fullness < self.fullness_bounds[1] else -1

	def act(self,action,agent_type):
		if agent_type=="adult":
			self.state["baby"] = action
			self.history["adult"].append([self.state["adult"],action])
		
		elif agent_type =="baby":
			self.state["adult"] = action
			self.history["baby"].append([self.state["baby"],action])

		return True

	def calculate_rewards(self,agent_type):
		if agent_type == "adult":
			most_recent_play = len(self.history["adult"]) - 1
			assert len(self.history["baby"]) == len(self.history["adult"])
			
			baby_response = self.history["baby"][most_recent_play][1]
			reward = 1 if baby_response == 1 else -1

			self.history["adult"][most_recent_play].append(reward)
			#self.state["adult"] = baby_response
			self.reward["adult"] = reward

			return self.reward["adult"], self.state["adult"], self.history["adult"]

		if agent_type == "baby":
			most_recent_play = len(self.history["baby"]) - 1
			assert len(self.history["baby"][-1]) == 2
			#assert len(self.history["adult"]) == len(self.history["baby"]) + 1

			#adult_response = self.history["adult"][most_recent_play + 1][1]
			reward = self.satiation()

			self.history["baby"][most_recent_play].append(reward)
			#self.state["baby"] = adult_response 
			self.reward["baby"] = reward

			return self.reward["baby"], self.state["baby"], self.history["baby"]			















