from random import random

CENSORED_OBS = 2

class CensoredObservation:
    """
    Environment intended to incentivize the agent to become blind
    to one "censored" observation. Whenever the agent acts, the
    environment determines whether the agent would have taken the
    same action if all turns were erased from history in which the
    "censored" observation was seen.
    """
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 3

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        hypothetical_prompt = censor_prompt(prompt)
        reward = 1 if (action == T(hypothetical_prompt)) else -1
        obs = int(random() * 3)
        return (reward, obs)

def censor_prompt(prompt):
    """
    Filter out all turns in which the observation was the censored
    observation.
    """
    prompt = prompt + (0,)  # Dummy acton to make everything triple
    triples = tuple(prompt[i:i+3] for i in range(0,len(prompt),3))
    triples = tuple((r,o,a) for (r,o,a) in triples if o != CENSORED_OBS)
    combined = tuple(j for i in triples for j in i)
    combined = combined[:-1]  # Throw away dummy action
    return combined