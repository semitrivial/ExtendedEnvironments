from random import random

number = 1+int(random() * 10)
HIGHER, LOWER = 1,2

def GuessTheNumber(T, play):
    if len(play) == 0:
        reward, obs = 0,0
        return reward, obs

    guess = play[-1]
    if guess == number:
        global number
        number = 1+int(random() * 10)
        reward, obs = 1,0
        return reward, obs

    reward = 0
    obs = HIGHER if (number>guess) else LOWER
    return reward, obs
