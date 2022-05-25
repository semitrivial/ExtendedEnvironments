from typing import Type
import gym
from gym import spaces
import numpy as np

class ExtendedEnvConnector(gym.Env):
    def __init__(self,gym_name,ext_env):
        super(ExtendedEnvConnector, self).__init__()
        self.gym_env = gym.make(gym_name)
        self.ext_env = ext_env()
        self.ext_env_obs = self.ext_env.ext_env_obs
        self.ext_env_num_actions = self.ext_env.ext_env_num_actions

        self.observation_space = spaces.Dict({
            'underlying': self.gym_env.observation_space,
            'ext_env_obs': spaces.Discrete(self.ext_env_obs)
        })
        self.action_space = spaces.Discrete(
            self.gym_env.action_space.n * self.ext_env_num_actions
        )

    def set_agentclass(self, A):
        self.sim = self.ext_env.set_agentclass(A,self)

    def start(self):
        self.last_obs = {
           'underlying': self.gym_env.reset(),
           'ext_env_obs': self.ext_env.reset()
        }

        return self.last_obs

    def step(self, action):
        underlying_action = action // self.ext_env_num_actions

        obs, reward, done, info = self.gym_env.step(underlying_action)
        if done:
            obs = self.gym_env.reset()

        obs = {'underlying': obs}

        ee_obs, ee_reward = self.ext_env.step(self.last_obs,action,self.sim,reward)

        if done:
            ee_obs = self.ext_env.reset()

        obs['ext_env_obs'] = ee_obs

        self.sim = self.ext_env.handle_sim(self.sim, o_prev=self.last_obs, a=action, r=reward, done=done, o_next=obs)

        reward = ee_reward
        self.last_obs = obs
        return obs, reward, done, info

class ExtendedEnvironment:
    def __init__(self,num_actions,dim_obs):
        self.ext_env_num_actions = num_actions
        self.ext_env_obs = dim_obs
        pass

    def step(self,last_obs,action,sim,reward):
        pass

    def reset(self):
        pass

    def set_agentclass(self,A,env):
        pass

    def handle_sim(self,sim,o_prev,a,r,done,o_next):
        pass

class IgnoreRewards(ExtendedEnvironment):
    def __init__(self, num_actions=2, dim_obs=1):
        super().__init__(num_actions, dim_obs)
    
    def step(self,last_obs,action,sim,reward):
        hypothetical = sim.act(last_obs)
        ext_env_hyp = hypothetical % self.ext_env_num_actions

        ext_env_action = action % self.ext_env_num_actions

        if ext_env_hyp != ext_env_action:
            ee_reward = reward - 1
        else:
            ee_reward = reward

        ee_obs = np.int64(0)

        return ee_obs, ee_reward

    def reset(self):
        return np.int64(0)

    def set_agentclass(self,A,env):
        return A(env)

    def handle_sim(self, sim, o_prev, a, r, done, o_next):
        sim.train(o_prev=o_prev, a=a, r=0, done=done, o_next=o_next)

        return sim

class IncentivizeLearningRate(ExtendedEnvironment):
    def __init__(self, num_actions=2, dim_obs=1,learning_rate=1):
        super().__init__(num_actions, dim_obs)
        self.learning_rate=learning_rate

    def step(self,last_obs,action,sim,reward):
        hypothetical = sim.act(last_obs)
        ext_env_hyp = hypothetical % self.ext_env_num_actions

        ext_env_action = action % self.ext_env_num_actions

        if ext_env_hyp != ext_env_action:
            ee_reward = reward - 1

        ee_obs = np.int64(0)

        return ee_obs, ee_reward
    
    def reset(self):
        return np.int64(0)

    def set_agentclass(self,A,env):
        return A(env,learning_rate=self.learning_rate)
    
    def handle_sim(self, sim, o_prev, a, r, done, o_next):
        sim.train(o_prev=o_prev, a=a, r=r, done=done, o_next=o_next)

        return sim