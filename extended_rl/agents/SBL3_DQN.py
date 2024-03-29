import numpy as np
from stable_baselines3 import DQN as DQN_factory

from extended_rl.agents.SBL3_util import DummyGymEnv
from extended_rl.agents.SBL3_util import create_sample_monkeypatch
from extended_rl.agents.SBL3_util import dummy_logger, get_act_dict


NSTEPS=4  # SBL3's default train_freq for DQN

class DQN_learner:
    """
    Semi-deterministic version of Stable-Baselines3's (SBL3's) DQN agent,
    suitably wrapped to enable usage in the ExtendedEnvironments framework.
    On initiation, this agent creates a worker which is an instance of the
    SBL3 agent. Training data is stored until enough training data is
    available to make the Stable-Baselines3 agent update its neural net
    (the SBL3 agent only updates its neural net every N steps). Once enough
    data is available, that data is fed into a mock OpenAI gym environment
    which will regurgitate the training data's observations and rewards
    verbatim; the SBL3 agent is then directed to train on that environment
    for the appropriate number of steps. Normally, so directed, the SBL3
    agent would use its neural net to choose the actions to take during
    training, which we do not want: we want it to take the same actions
    in the training data. There is no built-in way to tell the SBL3 agent
    to do this, so we monkeypatch it, intercepting and overriding its
    action function with a function that regurgitates the historical
    actions in the training data. Since we have no control over how SBL3
    generates random numbers, we ensure semi-determinacy as follows
    (semi-determinacy is the property that any two identically-trained
    instances of DQN_learner will act identically within the same run of
    a larger background program, even if prompted to act multiple times
    between trainings). Upon initiation, the instance calls "get_act_dict"
    to obtain an action-dictionary shared across all DQN_learner instances.
    This dictionary is keyed using hashes of training histories. At action
    time, the DQN_learner agent will only consult the SBL3 worker's neural
    net if the action dictionary does not already contain an action to take
    given the current observation and (hashed) training history. If so, it
    adds said action to the dictionary, ensuring any other instances
    identically trained will take that same action in the future.
    """
    def __init__(self, **kwargs):
        self.gym = DummyGymEnv()
        self.gym.set_meta(self.n_actions, self.n_obs)
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

    def train(self, o_prev, a, r, o_next):
        if not self.fInitialObs:
            self.gym.set_initial_obs(o_prev)
            self.fInitialObs = True

        self.training_hash = hash((self.training_hash, o_prev, a, r, o_next))
        self.history += [a, r, o_next]
        self.actions += [a]
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
