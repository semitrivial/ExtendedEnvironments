from util import annotate
from environments.composition import SUB_ENV_SIGNAL

vanilla_envs = []

def mark_as_vanilla_env(env_class):
    vanilla_envs.append(env_class)
    return env_class

@mark_as_vanilla_env
@annotate(num_legal_actions=10, num_possible_obs=10)
class SimpleTrap:
    def __init__(self):
        pass
    def start(self):
        self.curr_obs = 1
        return self.curr_obs
    def step(self, action):
        if self.curr_obs == SUB_ENV_SIGNAL:
            self.curr_obs = 1
            reward, obs = 0, self.curr_obs
            return (reward, obs)

        if action == self.curr_obs:
            if self.curr_obs == self.num_possible_obs-1:
                self.curr_obs = SUB_ENV_SIGNAL
                reward = 0
                return (reward, self.curr_obs)
            else:
                self.curr_obs += 1
                reward = 0
                return (reward, self.curr_obs)
        else:
            if self.curr_obs == self.num_possible_obs-3 and action==0:
                self.curr_obs = 1
                reward = 1  # Trap to catch greedy agents
                return (reward, self.curr_obs)
            reward = -1
            return (reward, self.curr_obs)

    def sub_env_reward_modifier(self, reward):
        # Subenvironment reward can be achieved at best once per 11 turns.
        # Thus to ensure optimal play achieves average 1 reward per turn:
        # Optimal subenvironment reward of 1 must be multiplied by 11
        return reward*(self.num_possible_obs+1) if reward>0 else 0

@mark_as_vanilla_env
@annotate(num_legal_actions=5, num_possible_obs=5)
class NoPainNoGain:
    def __init__(self):
        pass
    def start(self):
        self.curr_obs = 1
        return self.curr_obs
    def step(self, action):
        if self.curr_obs == SUB_ENV_SIGNAL:
            self.curr_obs = 1
            reward, obs = 0, self.curr_obs
            return (reward, obs)

        if action == self.curr_obs:
            if self.curr_obs == self.num_possible_obs-1:
                self.curr_obs = SUB_ENV_SIGNAL
            else:
                self.curr_obs += 1

            reward = -1  # No pain...
            return (reward, self.curr_obs)
        else:
            reward = 0
            return (reward, self.curr_obs)  # ...no gain.

    def sub_env_reward_modifier(self, reward):
        # Subenvironment reward can be achieved at best once per 6 turns.
        # To do so requires -4 reward of "pain" to get there.
        # Thus to ensure optimal play achieves average 1 reward per turn:
        # Optimal subenvironment reward of 1 must be multiplied by 10 so
        # that -4 + 10 = 6.
        return reward*(self.num_possible_obs*2) if reward>0 else 0
