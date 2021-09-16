import numpy as np
from gym import Env, spaces


# Utilities for SBL3_A2C.py, SBL3_DQN.py and SBL3_PPO.py.

class DummyGymEnv(Env):
    """
    Dummy OpenAI Gym environment which regurgitates historical percepts
    so that a Stable-Baselines3 agent can train on that history.
    """
    def __init__(self):
        super(DummyGymEnv, self).__init__()

    def set_meta(self, n_actions, n_obs):
        self.action_space = spaces.Discrete(n_actions)
        self.observation_space = spaces.Discrete(n_obs)

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
    """
    Stable-Baselines3 (SBL3) agents have no way of telling them which
    actions to take when you tell them to go train in an environment.
    We need to have train SBL3 agents on a mock environment which
    regurgitates historical percepts, and we need the SBL3 agents
    themselves to regurgitate historical actions while training there.
    Otherwise, the SBL3 agent would re-compute actions, possibly going
    out of sync with the history the mock environment is regurgitating.
    This function creates a monkeypatch-function for intercepting and
    overriding the SBL3 PPO or A2C action-function. (A different
    monkeypatch is needed for DQN since it uses a different action-
    function).
    """
    def forward_monkeypatch(*args):
        _actions, values, log_probs = A.worker_forward(*args)
        assert len(_actions) == 1
        _actions[0] = A.actions[A.worker.num_timesteps % n_steps]
        return _actions, values, log_probs

    return forward_monkeypatch

def create_sample_monkeypatch(A, n_steps):
    """
    Stable-Baselines3 (SBL3) agents have no way of telling them which
    actions to take when you tell them to go train in an environment.
    We need to have train SBL3 agents on a mock environment which
    regurgitates historical percepts, and we need the SBL3 agents
    themselves to regurgitate historical actions while training there.
    Otherwise, the SBL3 agent would re-compute actions, possibly going
    out of sync with the history the mock environment is regurgitating.
    This function creates a monkeypatch-function for intercepting and
    overriding the SBL3 DQN action-function. (A different monkeypatch
    is needed for PPO or A2C since those use a different action-function
    than DQN).
    """
    def sample_monkeypatch(*args):
        action = np.array([A.actions[A.worker.num_timesteps % n_steps]])
        return action, action

    return sample_monkeypatch

class DummyLogger:
    """
    Logger class whose instances silently ignore instructions to log
    things. This is used to gag Stable-Baselines3 log-writing which
    would otherwise waste precious time.
    """
    @staticmethod
    def record(*args, **kwargs):
        pass
    @staticmethod
    def dump(*args, **kwargs):
        pass

dummy_logger = DummyLogger()

act_dicts = {}

def get_act_dict(A, family, hyperparams_dict):
    """
    Create an action-dictionary to be shared across multiple instances of a
    one of our SBL3-based agent classes. Since we have no control over how
    SBL3 agents generate random numbers, we use these shared dictionaries to
    ensure semi-determinacy (semi-determinacy is the property that two
    instances of an agent-class, if instantiated within the same run of a
    larger background program, will act identically if they are trained
    identically). A given instance of one of our SBL3-based agent classes
    will only consult its underlying SBL3 worker's neural net if it does not
    find the appropriate observation/training-history-hash in the appropriate
    shared dictionary. If so, after generating an action using that neural
    net, the agent will add that action into the dictionary so other instances
    like itself, if identically trained, will re-use that action.
    """
    hyperparam_keys = hyperparams_dict.keys()
    hyperparams = [hyperparams_dict[k] for k in hyperparam_keys]
    hyperparams.sort()
    hyperparams = tuple(hyperparams)

    key = (family, hyperparams, A.n_actions, A.n_obs)
    if key in act_dicts:
        return act_dicts[key]
    else:
        act_dicts[key] = {}
        return act_dicts[key]
