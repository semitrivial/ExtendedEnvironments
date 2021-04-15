from random import random

def bandit1_reward_fnc(lever):
    return lever

class Bandit1:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.max_reward_per_action = 1
        self.min_reward_per_action = 0
        self.fnc = lambda T, play: abstract_bandit_env(T, play, bandit1_reward_fnc)

def bandit2_reward_fnc(lever):
    return lever

class Bandit2:
    def __init__(self):
        self.num_legal_actions = 3
        self.num_possible_obs = 1
        self.max_reward_per_action = 2
        self.min_reward_per_action = 0
        self.fnc = lambda T, play: abstract_bandit_env(T, play, bandit2_reward_fnc)

def bandit3_reward_fnc(lever):
    return 1+lever

class Bandit3:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.max_reward_per_action = 2
        self.min_reward_per_action = 1
        self.fnc = lambda T, play: abstract_bandit_env(T, play, bandit3_reward_fnc)

def bandit4_reward_fnc(lever):
    return 1+lever

class Bandit4:
    def __init__(self):
        self.num_legal_actions = 3
        self.num_possible_obs = 1
        self.max_reward_per_action = 3
        self.min_reward_per_action = 1
        self.fnc = lambda T, play: abstract_bandit_env(T, play, bandit4_reward_fnc)

def bandit5_reward_fnc(lever):
    if lever == 0:
        return 2

    return int(random()*4)

class Bandit5:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.max_reward_per_action = 3
        self.min_reward_per_action = 0
        self.fnc = lambda T, play: abstract_bandit_env(T, play, bandit5_reward_fnc)

def abstract_bandit_env(T, play, reward_fnc):
    if len(play) == 0:
        reward, obs = 0, 0
        return reward, obs

    lever = play[-1]
    reward = reward_fnc(lever)
    obs = 0
    return reward, obs