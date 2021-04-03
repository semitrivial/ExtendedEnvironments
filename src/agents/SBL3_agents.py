from math import log2

import stable_baselines3 as SBL3
from gym import Env, spaces
import numpy as np

class DummyEnv(Env):
    def __init__(self):
        super(DummyEnv, self).__init__()
    def set_meta(self, num_legal_actions, num_possible_obs):
        if isinstance(num_legal_actions, int):
            self.action_space = spaces.Discrete(num_legal_actions)
        else:
            self.action_space = num_legal_actions

        if isinstance(num_possible_obs, int):
            self.observation_space = spaces.Discrete(num_possible_obs)
        else:
            self.observation_space = num_possible_obs
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

        if not((str(train_on), str(meta)) in cache):
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
            cache[(str(train_on), str(meta))] = A
        else:
            A = cache[(str(train_on), str(meta))]

        if str(meta) == '(2, MultiDiscrete([3 2 2 3 2 2 3 2 2 3 2]))':
            if isinstance(prompt[-1], int) or type(prompt[-1]) == type(np.int64(0)):
                history = prompt[-11:]
                if len(history) < 11:
                    history = tuple([0]*(11-len(history))) + history
                history = list(history)
                history[0] += 1
                history[3] += 1
                history[6] += 1
                history[9] += 1
                history = tuple(history)
                try:
                    action, _ = A.predict(history)
                except Exception:
                    try:
                        action, _ = A.predict(history)
                    except Exception:
                        action, _ = A.predict(history)
            else:
                action, _ = A.predict(prompt[-1])
        else:
            action, _ = A.predict(prompt[-1])

        return action

    return agent

agent_A2C = SBL_agent(SBL3.A2C)
agent_PPO = SBL_agent(SBL3.PPO)
agent_DQN = SBL_agent(SBL3.DQN)
agent_DQN.requires_numpy_transl = True