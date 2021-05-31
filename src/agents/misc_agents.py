import random

from util import memoize

@memoize
def random_agent(prompt, num_legal_actions, num_possible_obs):
    """
    Agent which blindly outputs a random action (however, this
    agent is memoized, so its random response to a given prompt
    will not change if the same prompt is later input to the
    agent a second time)
    """
    return int(random.random() * num_legal_actions)

def constant_agent(prompt, num_legal_actions, num_possible_obs):
    """
    Agent which always outputs action 0.
    """
    return 0