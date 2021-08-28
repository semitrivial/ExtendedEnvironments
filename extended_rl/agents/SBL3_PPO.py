from stable_baselines3 import PPO as PPO_factory

from agents.SBL3_util import DummyGymEnv, create_fwd_monkeypatch


NSTEPS=2048  # SBL3's default n_steps for PPO

class PPO_learner:
    def __init__(self, **kwargs):
        self.gym = DummyGymEnv()
        self.gym.set_meta(self.num_legal_actions, self.num_possible_obs)
        self.worker = PPO_factory('MlpPolicy', self.gym, **kwargs)
        self.worker_forward = self.worker.policy.forward
        self.actions = []
        self.history = []
        self.monkeypatch = create_fwd_monkeypatch(self, NSTEPS)
        self.fInitialObs = False
        self.training_cnt = 0

    def act(self, obs):
        action, _ = self.worker.predict(obs)
        return action

    def train(self, o_prev, act, R, o_next):
        if not self.fInitialObs:
            self.gym.set_initial_obs(o_prev)
            self.fInitialObs = True

        self.history += [act, R, o_next]
        self.actions += [act]
        self.training_cnt += 1

        if self.training_cnt == NSTEPS:
            self.training_cnt = 0
            self.gym.set_history(self.history)
            self.worker.policy.forward = self.monkeypatch
            self.worker.learn(NSTEPS)
            self.worker.policy.forward = self.worker_forward
            self.history = []
            self.actions = []            
