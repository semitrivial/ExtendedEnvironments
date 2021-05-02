class DelayReactions:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.fnc = delay_reactions

def delay_reactions(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return (reward, obs)

    prompt, action = play[:-1], play[-1]
    hypothetical_prompt = shift_rewards(prompt)
    reward = 1 if (action == T(hypothetical_prompt)) else -1
    obs = 0
    return (reward, obs)

def shift_rewards(prompt):
    prompt = list(prompt)
    for i in range(len(prompt)-2, 0, 3):
        prompt[i] = prompt[i-3]
    return tuple(prompt)