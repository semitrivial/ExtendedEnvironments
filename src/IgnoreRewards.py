class IgnoreRewards:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        hypothetical_prompt = strip_rewards(prompt)
        reward = 1 if (action == T(hypothetical_prompt)) else -1
        obs = 0
        return (reward, obs)

def strip_rewards(prompt):
    prompt = prompt + (0,)  # Dummy action to make everything triple
    triples = tuple(prompt[i:i+3] for i in range(0,len(prompt),3))
    triples = tuple((0,o,a) for (r,o,a) in triples)
    combined = tuple(j for i in triples for j in i)
    combined = combined[:-1]  # Throw away dummy action
    return combined