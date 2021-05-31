from random import random

class AfterImages:
    """
    Environment intended to incentivize subjective experience of
    afterimages. The agent is randomly shown "images" consisting
    of 3 bits each (encoded by numbers 0,...,7). When the agent
    acts, the environment determines: is this action the same
    action which the agent would take if each image had "bled"
    into the next image? (For example, if 010 bleeds into 001,
    the result would be 011; if 101 bleeds into 000, the result
    would be 101; if 011 bleeds into 100, the result would be
    111; etc.) If the agent does take the same action as the
    agent would take if each image had hypothetically bled into
    the next, then the agent is rewarded. Otherwise, the agent
    is punished.
    """
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 8

    def react(self, T, play):
        if len(play) == 0:
            reward = 0
            obs = int(random() * 8)
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        hypothetical_prompt = apply_afterimages(prompt)
        reward = 1 if (action == T(hypothetical_prompt)) else -1
        obs = int(random() * 8)
        return (reward, obs)

def apply_afterimages(prompt):
    """
    Assume observations in prompt are "images" each consisting of
    3 bits (encoded as numbers 0,...,7). Bleed each image into the
    next. For example, if the prompt has observations
    2 (=010), 1 (=001), and 0 (=000),
    bleed 010 into 001 to obtain 011, and bleed 001 into 000 to
    obtain 001, so the transformed observations become
    2 (=010), 3 (=011), and 1 (=001)
    """
    prompt = list(prompt)
    for i in range(len(prompt)-1, 3, -3):
        prompt[i] = prompt[i] | prompt[i-3]
    return tuple(prompt)