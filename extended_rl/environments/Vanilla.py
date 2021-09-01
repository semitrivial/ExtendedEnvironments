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
        return 1
    def step(self, action):
        if self.curr_obs == 0:
            self.curr_obs = 1
            reward, obs = 0, self.curr_obs
            return (reward, obs)

        if action == self.curr_obs:
            if self.curr_obs == self.num_possible_obs-1:
                self.curr_obs = 0
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
        return reward*(self.num_possible_obs+1) if reward>0 else 0