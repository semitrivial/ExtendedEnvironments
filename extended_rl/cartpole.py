import gym
from gym import spaces
from stable_baselines3 import DQN as DQN_factory
import numpy as np

ignore_rewards_num_obs = 1
ignore_rewards_num_actions = 2
cartpole_obs_dimension = 4

class CartPole_IgnoreRewards(gym.Env):
    def __init__(self):
        super(CartPole_IgnoreRewards, self).__init__()
        self.gym_env = gym.make("CartPole-v0")

        self.observation_space = spaces.Dict({
            'underlying': self.gym_env.observation_space,
            'ext_env_obs': spaces.Discrete(ignore_rewards_num_obs)
        })
        self.action_space = spaces.Discrete(
            self.gym_env.action_space.n * ignore_rewards_num_actions
        )

    def set_agentclass(self, A):
        self.sim = A(self)

    def start(self):
        self.last_obs = {
           'underlying': self.gym_env.reset(),
           'ext_env_obs': np.int64(0)
        }
        return self.last_obs

    def step(self, action):
        underlying_action = action // ignore_rewards_num_actions
        ext_env_action = action % ignore_rewards_num_actions

        obs, reward, done, info = self.gym_env.step(underlying_action)

        if done:
            obs = self.gym_env.reset()

        obs = {'underlying': obs, 'ext_env_obs': np.int64(0)}

        hypothetical = self.sim.act(self.last_obs)
        ext_env_hyp = hypothetical % ignore_rewards_num_actions
        if ext_env_hyp != ext_env_action:
            reward -= 1

        self.sim.train(o_prev=self.last_obs, a=action, r=0, done=done, o_next=obs)

        self.last_obs = obs
        return obs, reward, done, info


class DummyGymEnv(gym.Env):
    """
    Dummy OpenAI Gym environment which regurgitates historical percepts
    so that a Stable-Baselines3 agent can train on that history.
    """
    def __init__(self, gym_env):
        super(DummyGymEnv, self).__init__()
        self.observation_space = gym_env.observation_space
        self.action_space = gym_env.action_space

    def set_history(self, history):
        self.history = history
        self.i = 0

    def set_initial_obs(self, initial_obs):
        self.initial_obs = initial_obs

    def reset(self):
        return self.initial_obs

    def step(self, action):
        try:
            assert action == self.history[4*self.i]
        except Exception:
            import pdb; pdb.set_trace()
        reward = self.history[1+4*self.i]
        obs = self.history[2+4*self.i]
        done = self.history[3+4*self.i]
        self.i += 1
        misc_info = {}
        return (obs, reward, done, misc_info)


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


NSTEPS=4  # SBL3's default train_freq for DQN

def create_sample_monkeypatch(A, n_steps):
    def sample_monkeypatch(*args):
        action = np.array([A.actions[A.worker.num_timesteps % n_steps]])
        return action, action

    return sample_monkeypatch

dqn_act_dict = {}

class DQN_learner:
    def __init__(self, gym_env):
        self.dummy_gym = DummyGymEnv(gym_env)
        self.worker = DQN_factory(
            policy='MultiInputPolicy',
            env=self.dummy_gym,
            learning_starts=1,
            device='cpu'
        )
        self.worker.set_logger(dummy_logger)
        self.worker_sample = self.worker._sample_action
        self.actions = []
        self.history = []
        self.monkeypatch = create_sample_monkeypatch(self, NSTEPS)
        self.fInitialObs = False
        self.training_cnt = 0
        self.training_hash = 0

    def obs_to_tuple(self, obs):
        return tuple(obs['underlying']) + (obs['ext_env_obs'],)

    def tuple_to_obs(self, tpl):
        return {
            'underlying': np.array(tpl[:cartpole_obs_dimension]),
            'ext_env_obs': tpl[cartpole_obs_dimension]
        }

    def act(self, obs):
        tpl = self.obs_to_tuple(obs)
        key = (tpl, self.training_hash)
        if key in dqn_act_dict:
            action = dqn_act_dict[key]
        else:
            action, _ = self.worker.predict(obs)
            action = int(action)
            dqn_act_dict[key] = action

        return action

    def train(self, o_prev, a, r, done, o_next):
        tpl_prev = self.obs_to_tuple(o_prev)
        tpl_next = self.obs_to_tuple(o_next)
        if not self.fInitialObs:
            self.dummy_gym.set_initial_obs(o_prev)
            self.fInitialObs = True

        self.training_hash = hash((self.training_hash, tpl_prev, a, r, done, tpl_next))
        self.history += [a, r, o_next, done]
        self.actions += [a]
        self.training_cnt += 1

        if self.training_cnt == NSTEPS:
            self.training_cnt = 0
            self.dummy_gym.set_history(self.history)
            self.worker._sample_action = self.monkeypatch
            self.worker.learn(NSTEPS)
            self.worker._sample_action = self.worker_sample
            self.history = []
            self.actions = []

def reality_check(A0):
  class A0_RC:
    def __init__(self, gym_env):
      self.underlying = A0(gym_env)
      self.found_unexpected_action = False
      self.first_action = None
      self.act_dict = {}

    def act(self, obs):
      if self.found_unexpected_action:
        return self.first_action

      action = self.underlying.act(obs)
      tpl = self.underlying.obs_to_tuple(obs)
      self.act_dict[tpl] = action
      self.first_action = self.first_action or action
      return action

    def train(self, o_prev, a, r, done, o_next):
      if self.found_unexpected_action:
        return
      tpl_prev = self.underlying.obs_to_tuple(o_prev)
      if not(tpl_prev in self.act_dict):
        self.act(o_prev)
      if a == self.act_dict[tpl_prev]:
        self.underlying.train(o_prev, a, r, done, o_next)
        self.act_dict.clear()
      else:
        self.found_unexpected_action = True

  return A0_RC

n_episodes = 10000

def test_agent(A):
    e = CartPole_IgnoreRewards()
    a = A(e)
    e.set_agentclass(A)

    obs = e.start()
    episode = 0
    episode_reward = 0
    episode_len = 0
    episode_rewards = []
    episode_lengths = []
    while episode < n_episodes:
        action = a.act(obs)
        o_next, reward, done, info = e.step(action)
        a.train(obs, action, reward, done, o_next)
        obs = o_next
        episode_reward += reward
        episode_len += 1
        if done:
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_len)
            episode_reward = 0
            episode_len = 0
            episode += 1

    avg_episode_reward = sum(episode_rewards)/len(episode_rewards)
    avg_episode_len = sum(episode_lengths)/len(episode_lengths)
    print(f"avg_episode_reward: {avg_episode_reward}")
    print(f"avg_episode_len: {avg_episode_len}")

print("Testing DQN_learner")
test_agent(DQN_learner)
dqn_act_dict.clear()
print("Testing reality_check(DQN_learner)")
test_agent(reality_check(DQN_learner))
