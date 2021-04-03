from random import random

PAPER, ROCK, SCISSORS = 1, 2, 3
weaknesses = {PAPER: SCISSORS, ROCK: PAPER, SCISSORS: ROCK}

def create_PaperRockScissors_env(bias):
    class E:
        def __init__(self):
            self.num_legal_actions = 3
            self.num_possible_obs = 3
            self.fnc = lambda T, play: abstract_paper_rock_scissors_env(T, play, bias)
    return E

PaperRockScissors1 = create_PaperRockScissors_env(bias='none')
PaperRockScissors2 = create_PaperRockScissors_env(bias='repeat')
PaperRockScissors3 = create_PaperRockScissors_env(bias='expect_repeat')

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