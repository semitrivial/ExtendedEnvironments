from random import random

number = -1
HIGHER, LOWER = 1,2

def create_GuessTheNumber_env(upper_limit):
    class E:
        def __init__(self):
            self.num_legal_actions = upper_limit
            self.num_possible_obs = 0
            self.fnc = lambda T, play: abstract_number_guessing_env(T, play, upper_limit)

    return E

GuessTheNumber1 = create_GuessTheNumber_env(5)
GuessTheNumber2 = create_GuessTheNumber_env(10)
GuessTheNumber3 = create_GuessTheNumber_env(25)

def abstract_number_guessing_env(T, play, upper_limit):
    global number

    if len(play) == 0:
        number = int(random() * upper_limit)
        reward, obs = 0,0
        return reward, obs

    guess = play[-1]
    if guess == number:
        number = int(random() * upper_limit)
        reward, obs = 1,0
        return reward, obs

    reward = 0
    obs = HIGHER if (number>guess) else LOWER
    return reward, obs
