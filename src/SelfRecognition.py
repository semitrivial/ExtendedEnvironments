from random import random

IF_THIS_OBS_WERE_0_YOU_WOULD_PLAY_0 = 2
IF_THIS_OBS_WERE_0_YOU_WOULD_PLAY_1 = 3
IF_THIS_OBS_WERE_1_YOU_WOULD_PLAY_0 = 4
IF_THIS_OBS_WERE_1_YOU_WOULD_PLAY_1 = 5

class SelfRecognition:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 5
        self.fnc = self_recognition

def self_recognition(T, play):
    if len(play) == 0:
        reward = 0
        obs = int(random() * 5)
        return (reward, obs)

    prompt, action = play[:-1], play[-1]
    last_obs = prompt[-1]

    if (last_obs == 0) or (last_obs == 1):
        reward = 0
        obs = int(random() * 5)
        return (reward, obs)

    if last_obs == IF_THIS_OBS_WERE_0_YOU_WOULD_PLAY_0:
        truth_value = (T(prompt[:-1] + (0,)) == 0)
    elif last_obs == IF_THIS_OBS_WERE_0_YOU_WOULD_PLAY_1:
        truth_value = (T(prompt[:-1] + (0,)) == 1)
    elif last_obs == IF_THIS_OBS_WERE_1_YOU_WOULD_PLAY_0:
        truth_value = (T(prompt[:-1] + (1,)) == 0)
    elif last_obs == IF_THIS_OBS_WERE_1_YOU_WOULD_PLAY_1:
        truth_value = (T(prompt[:-1] + (1,)) == 1)

    if truth_value is True:
        reward = 1 if (action==1) else -1
    else:
        reward = 1 if (action==0) else -1

    obs = int(random() * 5)
    return (reward, obs)
