from util import eval_and_count_steps

class PunishSlowAgent:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.max_reward_per_action = 1
        self.min_reward_per_action = -1
        self.fnc = punish_slow_agent

def punish_slow_agent(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return reward, obs

    prompt, action = play[:-1], play[1]

    local_vars = {'T': T, 'prompt': prompt}
    _, stepcount = eval_and_count_steps('T(prompt)', local_vars)

    reward = 1 if stepcount < 15*len(prompt) else -1
    obs = 0

    return reward, obs

class PunishFastAgent:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.max_reward_per_action = 1
        self.min_reward_per_action = -1
        self.fnc = punish_fast_agent

def punish_fast_agent(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return reward, obs

    prompt, action = play[:-1], play[1]

    local_vars = {'T': T, 'prompt': prompt}
    _, stepcount = eval_and_count_steps('T(prompt)', local_vars)

    reward = 1 if stepcount > 15*len(prompt) else -1
    obs = 0

    return reward, obs
