import numpy as np
from stable_baselines3 import DQN as DQN_factory

from agents.SBL3_util import DummyGymEnv, create_sample_monkeypatch


NSTEPS=4  # SBL3's default train_freq for DQN

class DQN_learner:
    def __init__(self, env, **kwargs):
        self.gym = DummyGymEnv()
        self.gym.set_meta(env.num_legal_actions, env.num_possible_obs)
        self.worker = DQN_factory(
            'MlpPolicy',
            self.gym,
            learning_starts=1,
            **kwargs
        )
        self.worker_sample = self.worker._sample_action
        self.actions = []
        self.history = []
        self.monkeypatch = create_sample_monkeypatch(self, NSTEPS)
        self.fInitialObs = False
        self.training_cnt = 0

    def act(self, obs):
        action, _ = self.worker.predict(np.int64(obs))
        return int(action)

    def train(self, o_prev, act, R, o_next):
        if not self.fInitialObs:
            self.gym.set_initial_obs(o_prev)
            self.fInitialObs = True

        self.history += [act, R, o_next]
        self.actions += [act]
        self.training_cnt += 1

        if self.training_cnt == NSTEPS:
            self.training_cnt = 0
            self.history = np.int64(self.history)
            self.gym.set_history(self.history)
            self.worker._sample_action = self.monkeypatch
            self.worker.learn(NSTEPS)
            self.worker._sample_action = self.worker_sample
            self.history = []
            self.actions = []            
