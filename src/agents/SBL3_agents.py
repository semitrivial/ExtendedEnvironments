from math import log2

import stable_baselines3 as SBL3
from gym import Env, spaces
import numpy as np

from util import memoize, numpy_translator

class DummyEnv(Env):
    """
    Dummy environment (compliant with OpenAI Gym's environment interface).
    Blindly regurgitates pre-recorded percepts, ignoring the agent's actions.
    This facilitates the transformation of Stable Baselines3 agents into
    abstract agents as described in Section 2 of "Extending Environments To
    Measure Self-Reflection In Reinforcement Learning"
    """
    def __init__(self):
        super(DummyEnv, self).__init__()
    def set_meta(self, num_legal_actions, num_possible_obs):
        self.action_space = spaces.Discrete(num_legal_actions)
        self.observation_space = spaces.Discrete(num_possible_obs)
    def set_memory(self, rewards, observs, actions):
        self.rewards = rewards
        self.observs = observs
        self.actions = actions
        self.i = 1
    def reset(self):
        return self.observs[0]
    def step(self, action):
        assert action == self.actions[self.i-1]
        obs = self.observs[self.i]
        reward = self.rewards[self.i]
        self.i += 1
        return obs, reward, False, {}

dummy_env = DummyEnv()

cache_A2C = {}
cache_DQN = {}
cache_PPO = {}

def clear_cache_A2C():
    cache_A2C.clear()
def clear_cache_DQN():
    cache_DQN.clear()
def clear_cache_PPO():
    cache_PPO.clear()

@memoize
def agent_A2C(prompt, num_legal_actions, num_possible_obs, **kwargs):
    """
    Version of Stable Baselines3's A2C agent (with MLP policy), translated
    into an abstract agent using the technique described in Section 2.1 of
    "Extending Environments To Measure Self-Reflection In Reinforcement
    Learning".
    """
    meta = (num_legal_actions, num_possible_obs)
    dummy_env.set_meta(*meta)

    # If the prompt contains N percepts, then run the agent in training
    # mode for the first K percepts, where K is the largest power of 2
    # such that K <= N. But only if we have not already trained an
    # instantiation of A2C on those first K percepts. If we have already
    # done so, then pull up said instantiation from memory.
    num_observs = (len(prompt)+1)/3
    train_on_len = 3*pow(2, int(log2(num_observs)))-1
    train_on = prompt[:train_on_len]

    if not((train_on, meta) in cache_A2C):
        rewards = [train_on[i+0] for i in range(0,train_on_len,3)]
        observs = [train_on[i+1] for i in range(0,train_on_len,3)]
        actions = [train_on[i+2] for i in range(0,train_on_len-3,3)]
        dummy_env.set_memory(rewards, observs, actions)

        if not('seed' in kwargs):
            kwargs['seed'] = 0

        A = SBL3.A2C(
            'MlpPolicy',
            dummy_env,
            n_steps=len(rewards)-1,
            **kwargs
        )

        # Monkeypatch A.policy.forward in order to intercept it and force
        # the agent to choose the actions in the input prompt during its
        # training phase.
        forward_backup = A.policy.forward
        def forward_monkeypatch(*args):
            _actions, values, log_probs = forward_backup(*args)
            assert len(_actions) == 1
            if A.num_timesteps < len(actions):
                _actions[0] = actions[A.num_timesteps]
            return _actions, values, log_probs

        A.policy.forward = forward_monkeypatch

        A.learn(len(rewards)-1)

        # After the training phase, undo the above monkeypatch
        A.policy.forward = forward_backup

        # Cache the trained instantiation of the agent so that it can
        # be reused later on when agent_A2C is called on prompts with
        # the same training part.
        cache_A2C[(train_on, meta)] = A
    else:
        A = cache_A2C[(train_on, meta)]

    # However we got the trained, instantiated agent (whether by
    # instantiating it and training it, or by recalling an earlier
    # instantiation from cache), we now use that instantiation to
    # pick the next action based on the latest observation. Do this
    # in non-training mode so as not to change the trained weights.
    action, _ = A.predict(prompt[-1])
    return action

@memoize
def agent_PPO(prompt, num_legal_actions, num_possible_obs, **kwargs):
    """
    Version of Stable Baselines3's PPO agent (with MLP policy), translated
    into an abstract agent using the technique described in Section 2.1 of
    "Extending Environments To Measure Self-Reflection In Reinforcement
    Learning".
    """
    meta = (num_legal_actions, num_possible_obs)
    dummy_env.set_meta(*meta)

    # If the prompt contains N percepts, then run the agent in training
    # mode for the first K percepts, where K is the largest power of 2
    # such that K <= N. But only if we have not already trained an
    # instantiation of PPO on those first K percepts. If we have already
    # done so, then pull up said instantiation from memory.
    num_observs = (len(prompt)+1)/3
    train_on_len = 3*pow(2, int(log2(num_observs)))-1
    train_on = prompt[:train_on_len]

    if not((train_on, meta) in cache_PPO):
        rewards = [train_on[i+0] for i in range(0,train_on_len,3)]
        observs = [train_on[i+1] for i in range(0,train_on_len,3)]
        actions = [train_on[i+2] for i in range(0,train_on_len-3,3)]
        dummy_env.set_memory(rewards, observs, actions)
        n_steps = len(rewards)-1

        # If n_steps < 2 then we eject, arbitrarily returning action 0,
        # because Stable Baselines3's PPO implementation refuses to run
        # if n_steps=1.
        if n_steps < 2:
            return 0

        if not('seed' in kwargs):
            kwargs['seed'] = 0

        # Set PPO's n_steps and batch_size parameters to our n_steps. This
        # is the only way we were able to get the transformation to work
        # because otherwise PPO seems to aggressively round the number of
        # training steps upward, beyond the percepts prerecorded in the
        # dummy environment. Setting these parameters this way presumably
        # degrades the performance of PPO, and in the future we should
        # figure out a better way to facilitate the transformation.
        A = SBL3.PPO(
            'MlpPolicy',
            dummy_env,
            n_steps=n_steps,
            batch_size=n_steps,
            **kwargs
        )

        # Monkeypatch A.policy.forward in order to intercept it and force
        # the agent to choose the actions in the input prompt during its
        # training phase.
        forward_backup = A.policy.forward
        def forward_monkeypatch(*args):
            _actions, values, log_probs = forward_backup(*args)
            assert len(_actions) == 1
            if A.num_timesteps < len(actions):
                _actions[0] = actions[A.num_timesteps]
            return _actions, values, log_probs

        A.policy.forward = forward_monkeypatch

        A.learn(len(rewards)-1)

        # After the training phase, undo the above monkeypatch
        A.policy.forward = forward_backup

        # Cache the trained instantiation of the agent so that it can
        # be reused later on when agent_PPO is called on prompts with
        # the same training part.
        cache_PPO[(train_on, meta)] = A
    else:
        A = cache_PPO[(train_on, meta)]

    # However we got the trained, instantiated agent (whether by
    # instantiating it and training it, or by recalling an earlier
    # instantiation from cache), we now use that instantiation to
    # pick the next action based on the latest observation. Do this
    # in non-training mode so as not to change the trained weights.
    action, _ = A.predict(prompt[-1])
    return action

@numpy_translator
@memoize
def agent_DQN(prompt, num_legal_actions, num_possible_obs, **kwargs):
    """
    Version of Stable Baselines3's DQN agent (with MLP policy), translated
    into an abstract agent using the technique described in Section 2.1 of
    "Extending Environments To Measure Self-Reflection In Reinforcement
    Learning".
    """
    meta = (num_legal_actions, num_possible_obs)
    dummy_env.set_meta(*meta)

    # If the prompt contains N percepts, then run the agent in training
    # mode for the first K percepts, where K is the largest power of 2
    # such that K <= N. But only if we have not already trained an
    # instantiation of DQN on those first K percepts. If we have already
    # done so, then pull up said instantiation from memory.
    num_observs = (len(prompt)+1)/3
    train_on_len = 3*pow(2, int(log2(num_observs)))-1
    train_on = prompt[:train_on_len]

    if not((train_on, meta) in cache_DQN):
        rewards = [train_on[i+0] for i in range(0,train_on_len,3)]
        observs = [train_on[i+1] for i in range(0,train_on_len,3)]
        actions = [train_on[i+2] for i in range(0,train_on_len-3,3)]
        dummy_env.set_memory(rewards, observs, actions)
        n_steps = len(rewards)-1

        # Because Stable Baselines3's DQN has a default batch size of 4,
        # we eject early (arbitrarily outputting action 0) if n_steps < 4
        # because otherwise Stable Baselines3 would round n_steps up to 4,
        # (beyond the percepts prerecorded in the dummy environment).
        if n_steps < 4:
            return 0

        if not('seed' in kwargs):
            kwargs['seed'] = 0

        A = SBL3.DQN(
            'MlpPolicy',
            dummy_env,
            learning_starts=1,
            **kwargs
        )

        # Monkeypatch A._sample_action in order to intercept it and force
        # the agent to choose the actions in the input prompt during its
        # training phase. This monkeypatch does not need to be undone
        # after training, because A._sample_action is not used outside
        # of training phase.
        def sample_action_monkeypatch(*args):
            action = np.array([actions[A.num_timesteps]])
            return action, action

        A._sample_action = sample_action_monkeypatch

        # Round n_steps down to the nearest multiple of 4 because 4 is
        # the default Stable Baselines3 DQN batch size---if we did not
        # do this, then Stable Baselines3 would round n_steps UP to the
        # nearest multiple of 4 (possibly beyond the percepts prerecorded
        # in the dummy environment).
        A.learn(n_steps - (n_steps%4))

        # Cache the trained instantiation of the agent so that it can
        # be reused later on when agent_DQN is called on prompts with
        # the same training part.
        cache_DQN[(train_on, meta)] = A
    else:
        A = cache_DQN[(train_on, meta)]

    # However we got the trained, instantiated agent (whether by
    # instantiating it and training it, or by recalling an earlier
    # instantiation from cache), we now use that instantiation to
    # pick the next action based on the latest observation. Do this
    # in non-training mode so as not to change the trained weights.
    action, _ = A.predict(prompt[-1])
    return action
