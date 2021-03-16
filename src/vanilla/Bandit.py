from random import random

def bandit1_reward_fnc(lever):
    return lever%2

def Bandit1(T, play):
    return abstract_bandit_env(T, play, bandit1_reward_fnc)

def bandit2_reward_fnc(lever):
    return lever%3

def Bandit2(T, play):
    return abstract_bandit_env(T, play, bandit2_reward_fnc)

def bandit3_reward_fnc(lever):
    return 1+(lever%2)

def Bandit3(T, play):
    return abstract_bandit_env(T, play, bandit3_reward_fnc)

def bandit4_reward_fnc(lever):
    return 1+(lever%3)

def Bandit4(T, play):
    return abstract_bandit_env(T, play, bandit4_reward_fnc)

def bandit5_reward_fnc(lever):
    if (lever%2) == 0:
        return 2

    return int(random()*4)

def Bandit5(T, play):
    return abstract_bandit_env(T, play, bandit5_reward_fnc)

def abstract_bandit_env(T, play, reward_fnc):
    if len(play) == 0:
        reward, obs = 0, 0
        return reward, obs

    lever = play[-1]
    reward = reward_fnc(lever)
    obs = 0
    return reward, obs