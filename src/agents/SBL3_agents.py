from math import log2

import stable_baselines3 as SBL3
from gym import Env, spaces

class DummyEnv(Env):
    def __init__(self):
        super(DummyEnv, self).__init__()
    def set_meta(self, num_legal_actions, num_possible_obs):
        self.action_space = spaces.Discrete(num_legal_actions)
        self.observation_space = spaces.Discrete(num_possible_obs)
    def set_rewards_and_observs(self, rewards, observs):
        self.rewards = rewards
        self.observs = observs
        self.i = 1
    def reset(self):
        return self.observs[0]
    def step(self, action):
        obs = self.observs[self.i]
        reward = self.rewards[self.i]
        self.i += 1
        return obs, reward, False, {}

dummy_env = DummyEnv()

caches = {
    SBL3.A2C: {},
    SBL3.DQN: {},
    SBL3.PPO: {},
}

def SBL_agent(learner):
    cache = caches[learner]

    def agent(prompt, num_legal_actions, num_possible_obs):
        dummy_env.set_meta(num_legal_actions, num_possible_obs)
        meta = (num_legal_actions, num_possible_obs)
        num_observs = (len(prompt)+1)/3
        train_on_len = 3*pow(2, int(log2(num_observs)))-1
        train_on = prompt[:train_on_len]

        if not((train_on, meta) in cache):
            rewards = [train_on[i+0] for i in range(0,train_on_len,3)]
            observs = [train_on[i+1] for i in range(0,train_on_len,3)]
            dummy_env.set_rewards_and_observs(rewards, observs)

            if learner == SBL3.PPO:
                n_steps = len(rewards)-1

                if n_steps < 2:
                    return 0

                A = learner('MlpPolicy', dummy_env, n_steps=n_steps, batch_size=n_steps)
            elif learner == SBL3.A2C:
                A = learner('MlpPolicy', dummy_env, n_steps=len(rewards)-1)
            else:
                A = learner('MlpPolicy', dummy_env, train_freq=len(rewards)-1)

            A.learn(len(rewards)-1)
            cache[(train_on, meta)] = A
        else:
            A = cache[(train_on, meta)]

        return A.predict(prompt[-1])

    return agent

agent_A2C = SBL_agent(SBL3.A2C)
agent_DQN = SBL_agent(SBL3.DQN)
agent_PPO = SBL_agent(SBL3.PPO)