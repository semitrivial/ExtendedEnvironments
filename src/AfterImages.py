from random import random

class AfterImages:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 8
        self.fnc = after_images

def after_images(T, play):
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
    prompt = list(prompt)
    for i in range(len(prompt)-1, 0, -3):
        prompt[i] = prompt[i] | prompt[i-3]
    return tuple(prompt)