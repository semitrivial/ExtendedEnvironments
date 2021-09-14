from stable_baselines3 import A2C as A2C_factory

from agents.SBL3_util import DummyGymEnv, create_fwd_monkeypatch
from agents.SBL3_util import dummy_logger, get_act_dict


NSTEPS=5  # SBL3's default n_steps for A2C

class A2C_learner:
    def __init__(self, **kwargs):
        self.gym = DummyGymEnv()
        self.gym.set_meta(self.num_legal_actions, self.num_possible_obs)
        self.worker = A2C_factory(
            policy='MlpPolicy',
            env=self.gym,
            device='cpu',
            **kwargs
        )
        self.worker.set_logger(dummy_logger)
        self.worker_forward = self.worker.policy.forward
        self.actions = []
        self.history = []
        self.monkeypatch = create_fwd_monkeypatch(self, NSTEPS)
        self.fInitialObs = False
        self.training_cnt = 0
        self.training_hash = 0
        self.act_dict = get_act_dict(self, A2C_learner, kwargs)

    def act(self, obs):
        key = (obs, self.training_hash)
        if key in self.act_dict:
            action = self.act_dict[key]
        else:
            action, _ = self.worker.predict(obs)
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
            self.gym.set_history(self.history)
            self.worker.policy.forward = self.monkeypatch
            self.worker.learn(NSTEPS)
            self.worker.policy.forward = self.worker_forward
            self.history = []
            self.actions = []            
