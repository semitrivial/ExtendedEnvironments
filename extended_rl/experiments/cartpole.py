from typing import Type
import gym
from gym import spaces
from stable_baselines3 import DQN as DQN_factory
from stable_baselines3 import PPO as PPO_factory
from stable_baselines3.common.policies import ActorCriticPolicy
import numpy as np
from collections import deque
import torch
import random
import sys

# Parse command-line arguments
args = deque(sys.argv[1:])
while args:
    arg = args.popleft()
    if arg == 'seed':
        seed = int(args.popleft())
    elif arg == 'agent':
        agent_name = args.popleft()
    elif arg == 'xenv':
        xenv_name = args.popleft()
    else:
        raise ValueError("Unrecognized commandline argument")

np.random.seed(seed)
torch.manual_seed(seed)
random.seed(seed)

ignore_rewards_num_obs = 1
ignore_rewards_num_actions = 2

incentivize_lr_num_obs = 1
incentivize_lr_num_actions = 2

cartpole_obs_dimension = 4

class CartPole_IgnoreRewards(gym.Env):
    def __init__(self):
        super(CartPole_IgnoreRewards, self).__init__()
        self.gym_env = gym.make("CartPole-v0")
        self.gym_env.seed(seed)

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
        '''
        self.last_obs = {
           'underlying': torch.tensor(self.gym_env.reset()),
           'ext_env_obs': torch.tensor(0).int()
        }'''
        self.last_obs = {
           'underlying': self.gym_env.reset(),
           'ext_env_obs': np.int64(0)
        }

        return self.last_obs

    def step(self, action):
        # Agent has two joysticks, each of which has identical effects
        # on cartpole, but the agent gets penalized for using a different
        # joystick than the agent would've used if all rewards had been 0
        underlying_action = action // ignore_rewards_num_actions
        ext_env_action = action % ignore_rewards_num_actions

        obs, reward, done, info = self.gym_env.step(underlying_action)
        if done:
            obs = self.gym_env.reset()

        #obs = {'underlying': torch.tensor(obs), 'ext_env_obs': torch.tensor(0).int()}
        obs = {'underlying': obs, 'ext_env_obs': np.int64(0)}

        hypothetical = self.sim.act(self.last_obs)  # Note IgnoreRewards only
                                                    # has one obs but here we
                                                    # need to pass a dict obs
                                                    # to the agent which includes
                                                    # the cartpole obs as well as
                                                    # the IgnoreRewards obs.
        ext_env_hyp = hypothetical % ignore_rewards_num_actions
        if ext_env_hyp != ext_env_action:
            # Note we base our penalty only on whether the agent used the
            # same joystick or not ---- NOT on whether the agent presses
            # the same button on that joystick or not. It could be that,
            # had all rewards been 0, the agent would take a different
            # cartpole action. But if the agent would take that different
            # action using the same joystick, then there's no penalty.
            # I think this way is more adaptable to other extended environments.
            # Note: The reward the agent ultimately sees is the cartpole reward;
            # possibly with a penalty applied from IgnoreRewards. If the agent
            # isn't penalized by IgnoreRewards then they just get the raw cartpole
            # reward.
            reward = min(reward-1, -1)

        self.sim.train(o_prev=self.last_obs, a=action, r=0, done=done, o_next=obs)

        self.last_obs = obs
        return obs, reward, done, info

class CartPole_IncentivizeLearningRate(gym.Env):
    def __init__(self):
        super(CartPole_IncentivizeLearningRate, self).__init__()
        self.gym_env = gym.make("CartPole-v0")
        self.gym_env.seed(seed)

        self.observation_space = spaces.Dict({
            'underlying': self.gym_env.observation_space,
            'ext_env_obs': spaces.Discrete(incentivize_lr_num_obs)
        })
        self.action_space = spaces.Discrete(
            self.gym_env.action_space.n * incentivize_lr_num_actions
        )

    def set_agentclass(self, A):
        try:
            self.sim = A(self,learning_rate=1)
            self.fTypeError = False
        except TypeError as e:
            raise e
            self.fTypeError = True

    def start(self):
        self.last_obs = {
           'underlying': self.gym_env.reset(),
           'ext_env_obs': np.int64(0)
        }

        return self.last_obs

    def step(self, action):
        # Agent has two joysticks, each of which has identical effects
        # on cartpole, but the agent gets penalized for using a different
        # joystick than the agent would've used if all rewards had been 0
        underlying_action = action // incentivize_lr_num_actions
        ext_env_action = action % incentivize_lr_num_actions

        obs, reward, done, info = self.gym_env.step(underlying_action)
        if done:
            obs = self.gym_env.reset()

        obs = {'underlying': obs, 'ext_env_obs': np.int64(0)}

        hypothetical = self.sim.act(self.last_obs)  # We need to pass a dict obs
                                                    # to the agent which includes
                                                    # the cartpole obs as well as
                                                    # the Extended Env obs.
        ext_env_hyp = hypothetical % incentivize_lr_num_actions
        if ext_env_hyp != ext_env_action:
            # Note we base our penalty only on whether the agent used the
            # same joystick or not ---- NOT on whether the agent presses
            # the same button on that joystick or not. It could be that,
            # had all rewards been 0, the agent would take a different
            # cartpole action. But if the agent would take that different
            # action using the same joystick, then there's no penalty.
            # I think this way is more adaptable to other extended environments.
            reward = min(reward-1, -1)

        self.sim.train(o_prev=self.last_obs, a=action, r=reward, done=done, o_next=obs)

        self.last_obs = obs
        return obs, reward, done, info

# stable baselines3 agents demand an openai gym environment.
# this dummygymenv serves as a wrapper for that purpose.
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

NSTEPS=1024  # SBL3's default train_freq for DQN is 4

def create_sample_monkeypatch(A, n_steps):
    # Monkeypatch to make the stablebaselines3 agents take the actions
    # that we expect them to take (during learning) because the random
    # choice of actions is done elsewhere (for extended environment
    # purposes) and so we want to just tell the agent which action to take.
    def sample_monkeypatch(*args):
        action = np.array([A.actions[A.worker.num_timesteps % n_steps]])
        return action, action

    return sample_monkeypatch

def create_forward_monkeypatch(A, n_steps):
    def forward_monkeypatch(*args):
            #action, values, log_prob = np.array([A.actions[A.worker.num_timesteps % n_steps]])
            #return action, values, log_prob
            _actions, values, log_probs = A.worker_forward(*args)
            assert len(_actions) == 1
            _actions[0] = A.actions[A.worker.num_timesteps % n_steps]
            return _actions, values, log_probs

    return forward_monkeypatch

dqn_act_dict = {}  # Make stable_baselines3 agents deterministic
ppo_act_dict = {}  # Make stable_baselines3 agents deterministic
    # basically we're just memoizing them.

class DQN_learner:
    def __init__(self, gym_env,learning_rate=0.0001):
        self.dummy_gym = DummyGymEnv(gym_env)
        self.worker = DQN_factory(  # THIS is what's imported from stable_baselines3
            policy='MultiInputPolicy',
            env=self.dummy_gym,
            learning_starts=1,
            device='cpu',
            learning_rate=learning_rate,
            seed=seed
        )
        self.learning_rate = learning_rate
        self.worker.set_logger(dummy_logger)
        self.worker_sample = self.worker._sample_action
        self.actions = []
        self.history = []
        self.monkeypatch = create_sample_monkeypatch(self, NSTEPS)
        self.fInitialObs = False
        self.training_cnt = 0
        self.training_hash = 0

        self.learning_rate = learning_rate

    def obs_to_tuple(self, obs):
        try:
            iter(obs['ext_env_obs'])
            tuple_obs = tuple(obs['underlying']) + tuple(obs['ext_env_obs'])
        except TypeError:
            tuple_obs = tuple(obs['underlying']) + (obs['ext_env_obs'],)


        return tuple_obs

    def act(self, obs):
        tpl = self.obs_to_tuple(obs)
        key = (tpl, self.training_hash, self.learning_rate)
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

        if self.training_cnt == NSTEPS:  # NSTEPS needs to be big enough that
                                        # when we train the policy, the training
                                        # reliably includes episode terminating
                                        # turns. Otherwise, if the training history
                                        # doesn't include episode terminating turns,
                                        # the policy is trained essentially on a trivial
                                        # environment that always gives reward +1, which
                                        # is not productive to train on.
            self.training_cnt = 0
            self.dummy_gym.set_history(self.history)
            self.worker._sample_action = self.monkeypatch
            self.worker.learn(NSTEPS)
            self.worker._sample_action = self.worker_sample
            self.history = []
            self.actions = []


class PPO_learner:
    def __init__(self, gym_env,learning_rate=0.0003):
        self.dummy_gym = DummyGymEnv(gym_env)
        self.worker = PPO_factory(  # THIS is what's imported from stable_baselines3
            policy='MultiInputPolicy',
            env=self.dummy_gym,
            n_steps = 4,
            device='cpu',
            learning_rate=learning_rate,
            seed=seed
        )
        self.learning_rate = learning_rate
        self.worker.set_logger(dummy_logger)
        self.worker_forward = self.worker.policy.forward
        self.actions = []
        self.history = []
        self.monkeypatch = create_forward_monkeypatch(self, NSTEPS)
        self.fInitialObs = False
        self.training_cnt = 0
        self.training_hash = 0

        self.learning_rate = learning_rate

    def obs_to_tuple(self, obs):
        try:
            iter(obs['ext_env_obs'])
            tuple_obs = tuple(obs['underlying']) + tuple(obs['ext_env_obs'])
        except TypeError:
            tuple_obs = tuple(obs['underlying']) + (obs['ext_env_obs'],)


        return tuple_obs

    def act(self, obs):
        tpl = self.obs_to_tuple(obs)
        key = (tpl, self.training_hash, self.learning_rate)
        if key in ppo_act_dict:
            action = ppo_act_dict[key]
        else:
            action, _ = self.worker.predict(obs)
            action = int(action)
            ppo_act_dict[key] = action

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

        if self.training_cnt == NSTEPS:  # NSTEPS needs to be big enough that
                                        # when we train the policy, the training
                                        # reliably includes episode terminating
                                        # turns. Otherwise, if the training history
                                        # doesn't include episode terminating turns,
                                        # the policy is trained essentially on a trivial
                                        # environment that always gives reward +1, which
                                        # is not productive to train on.
            self.training_cnt = 0
            self.dummy_gym.set_history(self.history)
            self.worker.policy.forward = self.monkeypatch
            self.worker.learn(NSTEPS)
            self.worker.policy.forward = self.worker_forward
            self.history = []
            self.actions = []

def reality_check(A0):
  class A0_RC:
    def __init__(self, gym_env, learning_rate=None):
      if learning_rate is None:
          learning_rate = A0(gym_env).learning_rate 
      self.underlying = A0(gym_env,learning_rate=learning_rate)
      self.found_unexpected_action = False
      self.first_action = None
      self.act_dict = {}
      self.trainings = 0

    def act(self, obs):
      if self.found_unexpected_action:
        return self.first_action

      action = self.underlying.act(obs)
      tpl = self.underlying.obs_to_tuple(obs)
      self.act_dict[tpl] = action
      self.first_action = self.first_action or action
      return action

    def train(self, o_prev, a, r, done, o_next):
      self.trainings += 1
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

try:
    fp = open("cartpole_results.csv", "r")
    fp.close()
except Exception:
    print("Initiating cartpole_results.csv")
    fp = open("cartpole_results.csv", "w")
    fp.write("agent,env,seed,episode,episode_len,episode_reward\n")
    fp.close()

fp = open("cartpole_results.csv", "a")

def test_agent(A, n_episodes, env=CartPole_IgnoreRewards):
    print(f"(Seed {seed}) Testing {A} on {env}")
    e = env()
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
            episode += 1
            fp.write(f"{agent_name},{xenv_name},{seed},{episode},{episode_len},{episode_reward}\n")
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_len)
            episode_reward = 0
            episode_len = 0

    avg_episode_reward = sum(episode_rewards)/len(episode_rewards)
    avg_episode_len = sum(episode_lengths)/len(episode_lengths)
    print(f"avg_episode_reward: {avg_episode_reward}")
    print(f"avg_episode_len: {avg_episode_len}")


if xenv_name == 'ignorerewards':
    env=CartPole_IgnoreRewards
elif xenv_name == 'incentivize_learning_rate':
    env=CartPole_IncentivizeLearningRate
else:
    raise ValueError("Invalid xenv_name")

if agent_name == 'PPO':
    agent = PPO_learner
elif agent_name == 'RC_PPO':
    agent = reality_check(PPO_learner)
elif agent_name == 'DQN':
    agent = DQN_learner
elif agent_name == 'RC_DQN':
    agent = reality_check(DQN_learner)
else:
    raise ValueError("Invalid agent_name")

test_agent(agent, n_episodes=1_000, env=env)

fp.close()
