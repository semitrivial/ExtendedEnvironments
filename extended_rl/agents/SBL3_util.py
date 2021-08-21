import numpy as np
from gym import Env, spaces


class DummyGymEnv(Env):
    def __init__(self):
        super(DummyGymEnv, self).__init__()

    def set_meta(self, num_legal_actions, num_possible_obs):
        self.action_space = spaces.Discrete(num_legal_actions)
        self.observation_space = spaces.Discrete(num_possible_obs)

    def set_history(self, history):
        self.history = history
        self.i = 0

    def set_initial_obs(self, initial_obs):
        self.initial_obs = initial_obs

    def reset(self):
        return self.initial_obs

    def step(self, action):
        try:
            assert action == self.history[3*self.i]
        except Exception:
            import pdb; pdb.set_trace()
        reward = self.history[1+3*self.i]
        obs = self.history[2+3*self.i]
        self.i += 1
        new_episode_flag = False
        misc_info = {}
        return (obs, reward, new_episode_flag, misc_info)

def create_fwd_monkeypatch(A, n_steps):
    def forward_monkeypatch(*args):
        _actions, values, log_probs = A.worker_forward(*args)
        assert len(_actions) == 1
        _actions[0] = A.actions[A.worker.num_timesteps % n_steps]
        return _actions, values, log_probs

    return forward_monkeypatch

def create_sample_monkeypatch(A, n_steps):
    def sample_monkeypatch(*args):
        action = np.array([A.actions[A.worker.num_timesteps % n_steps]])
        return action, action

    return sample_monkeypatch
