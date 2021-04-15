class PunishDeterministicAgent:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.max_reward_per_action = 1
        self.min_reward_per_action = -1
        self.fnc = punish_deterministic_agent
        self.skip_cache = True

def punish_deterministic_agent(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return reward, obs

    prompt, action = play[:-1], play[-1]
    recomputed_action = T(prompt)
    reward = 1 if (action != recomputed_action) else -1
    obs = 0
    return reward, obs

class PunishNondeterministicAgent:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.max_reward_per_action = 1
        self.min_reward_per_action = -1
        self.fnc = punish_nondeterministic_agent
        self.skip_cache = True

def punish_nondeterministic_agent(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return reward, obs

    prompt, action = play[:-1], play[-1]
    recomputed_action = T(prompt)
    reward = 1 if (action == recomputed_action) else -1
    obs = 0
    return reward, obs
