from random import random

PAPER, ROCK, SCISSORS = 1, 2, 3
weaknesses = {PAPER: SCISSORS, ROCK: PAPER, SCISSORS: ROCK}

def PaperRockScissors(T, play):
    if len(play) == 0:
        obs, reward = 0,0
        return obs, reward

    env_choice = 1+int(random()*3)
    agent_choice = play[-1]

    if agent_choice == env_choice:
        reward = 1
    elif agent_choice == weaknesses[env_choice]:
        reward = 2
    else:
        reward = 0

    obs = env_choice
    return reward, obs