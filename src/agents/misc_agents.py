import random

from util import memoize

@memoize
def random_agent(prompt, num_legal_actions, num_possible_obs):
    return int(random.random() * num_legal_actions)

def constant_agent(prompt, num_legal_actions, num_possible_obs):
    return 0