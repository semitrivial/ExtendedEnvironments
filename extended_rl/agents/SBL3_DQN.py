import numpy as np
from stable_baselines3 import DQN as DQN_factory

from agents.SBL3_util import DummyGymEnv, create_sample_monkeypatch
from agents.SBL3_util import dummy_logger, get_act_dict


NSTEPS=4  # SBL3's default train_freq for DQN

class DQN_learner:
    def __init__(self, **kwargs):
        self.gym = DummyGymEnv()
        self.gym.set_meta(self.num_legal_actions, self.num_possible_obs)
        self.worker = DQN_factory(
            policy='MlpPolicy',
            env=self.gym,
            learning_starts=1,
            device='cpu',
            **kwargs
        )
        self.worker.set_logger(dummy_logger)
        self.worker_sample = self.worker._sample_action
        self.actions = []
        self.history = []
        self.monkeypatch = create_sample_monkeypatch(self, NSTEPS)
        self.fInitialObs = False
        self.training_cnt = 0
        self.training_hash = 0
        self.act_dict = get_act_dict(self, DQN_learner, kwargs)

    def act(self, obs):
        key = (obs, self.training_hash)
        if key in self.act_dict:
            action = self.act_dict[key]
        else:
            action, _ = self.worker.predict(np.int64(obs))
            action = int(action)
            self.act_dict[key] = action

        return action

    def train(self, o_prev, act, R, o_next):
        if not self.fInitialObs:
            self.gym.set_initial_obs(o_prev)
            self.fInitialObs = True

        self.training_hash = hash((self.training_hash, o_prev, act, R, o_next))
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
