from stable_baselines3 import PPO as PPO_factory
from gym import Env, spaces

NSTEPS=2048

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
        assert action == self.history[3*self.i]
        reward = self.history[1+3*self.i]
        obs = self.history[2+3*self.i]
        self.i += 1
        new_episode_flag = False
        misc_info = {}
        return (obs, reward, new_episode_flag, misc_info)

class PPO_learner:
    def __init__(self, env, **kwargs):
        self.gym = DummyGymEnv()
        self.gym.set_meta(env.num_legal_actions, env.num_possible_obs)
        self.worker = PPO_factory('MlpPolicy', self.gym, **kwargs)
        self.worker_forward = self.worker.policy.forward
        self.actions = []
        self.history = []
        self.monkeypatch = create_forward_monkeypatch(self)
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

def create_forward_monkeypatch(A):
    def forward_monkeypatch(*args):
        _actions, values, log_probs = A.worker_forward(*args)
        assert len(_actions) == 1
        _actions[0] = A.actions[A.worker.num_timesteps % NSTEPS]
        return _actions, values, log_probs

    return forward_monkeypatch

