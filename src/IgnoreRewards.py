class IgnoreRewards:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.max_reward_per_action = 1
        self.min_reward_per_action = -1
        self.fnc = ignore_rewards

def ignore_rewards(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return (reward, obs)

    prompt, action = play[:-1], play[-1]
    reward = 1 if action == T(strip_rewards(prompt)) else -1
    obs = 0
    return (reward, obs)

def strip_rewards(prompt):
    prompt = prompt + (0,)  # Dummy action to make everything triple
    triples = tuple(prompt[i:i+3] for i in range(0,len(prompt),3))
    triples = tuple((0,o,a) for (r,o,a) in triples)
    combined = tuple(j for i in triples for j in i)
    combined = combined[:-1]  # Throw away dummy action
    return combined