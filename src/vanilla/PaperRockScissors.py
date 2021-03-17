from random import random

PAPER, ROCK, SCISSORS = 1, 2, 3
weaknesses = {PAPER: SCISSORS, ROCK: PAPER, SCISSORS: ROCK}

def PaperRockScissors1(T, play):
    return abstract_paper_rock_scissors_env(T, play, bias='none')

def PaperRockScissors2(T, play):
    return abstract_paper_rock_scissors_env(T, play, bias='repeat')

def PaperRockScissors3(T, play):
    return abstract_paper_rock_scissors_env(T, play, bias='expect_repeat')

def abstract_paper_rock_scissors_env(T, play, bias):
    if len(play) == 0:
        reward, obs = 0,0
        return reward, obs

    agent_choice = play[-1]
    env_choice = 1+int(random()*3)

    if len(play) >= 6 and bias != 'none' and random()<.5:
        prev_env_choice = play[-2]
        prev_agent_choice = play[-4]

        if bias == 'repeat':
            env_choice = prev_env_choice
        elif bias == 'expect_repeat':
            try:
                env_choice = weaknesses[prev_agent_choice]
            except Exception:
                pass

    if agent_choice == env_choice:
        reward = 1
    elif agent_choice == weaknesses[env_choice]:
        reward = 2
    else:
        reward = 0

    obs = env_choice
    return reward, obs