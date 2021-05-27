class AdversarialSequencePredictor:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 2

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        opposite_prompt = opposite_perspective(prompt)
        evader_action = T(opposite_prompt)
        reward = 1 if (action == evader_action) else -1
        obs = evader_action
        return (reward, obs)

class AdversarialSequenceEvader:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 2

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        opposite_prompt = opposite_perspective(prompt)
        evader_action = T(opposite_prompt)
        reward = 1 if (action != evader_action) else -1
        obs = evader_action
        return (reward, obs)

def opposite_perspective(prompt):
    prompt = prompt + (0,)  # Dummy action to make everything triple
    triples = tuple(prompt[i:i+3] for i in range(0,len(prompt),3))
    triples = tuple((-r,a,o) for (r,o,a) in triples)
    combined = tuple(j for i in triples for j in i)
    combined = combined[:-1]  # Throw away dummy action
    return combined