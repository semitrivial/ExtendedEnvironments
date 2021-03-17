from random import random

number = -1
HIGHER, LOWER = 1,2

def GuessTheNumber1(T, play):
    return abstract_number_guessing_env(T, play, upper_limit=5)

def GuessTheNumber2(T, play):
    return abstract_number_guessing_env(T, play, upper_limit=10)

def GuessTheNumber3(T, play):
    return abstract_number_guessing_env(T, play, upper_limit=25)

def abstract_number_guessing_env(T, play, upper_limit):
    if len(play) == 0:
        global number
        number = 1+int(random() * upper_limit)
        reward, obs = 0,0
        return reward, obs

    guess = play[-1]
    if guess == number:
        global number
        number = 1+int(random() * upper_limit)
        reward, obs = 1,0
        return reward, obs

    reward = 0
    obs = HIGHER if (number>guess) else LOWER
    return reward, obs
