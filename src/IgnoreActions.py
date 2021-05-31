from random import random

class IgnoreActions:
    """
    Environment which incentivizes the agent to ignore its own past actions.
    Whenever the agent acts, the environment determines: would the agent
    have acted the same if the past history preceding the action were
    identical except that the agent always took action 0? If so, then give
    the agent reward +1, otherwise, give the agent reward -1.
    """
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        reward = 1 if action == T(set_all_actions_to_0(prompt)) else -1
        obs = 0
        return (reward, obs)

def set_all_actions_to_0(prompt):
    prompt = prompt + (0,)  # Dummy action to make everything triple
    triples = tuple(prompt[i:i+3] for i in range(0,len(prompt),3))
    triples = tuple((r,o,0) for (r,o,a) in triples)
    combined = tuple(j for i in triples for j in i)
    combined = combined[:-1]  # Throw away dummy action
    return combined