class IgnoreRewards2:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.fnc = ignore_rewards2

def ignore_rewards2(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return (reward, obs)

    prompt, action = play[:-1], play[-1]
    hypothetical_prompt = [0]*len(prompt)
    reward = 1 if (action == T(hypothetical_prompt)) else -1
    obs = 0
    return (reward, obs)