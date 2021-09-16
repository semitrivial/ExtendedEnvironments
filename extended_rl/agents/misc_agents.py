from extended_rl.prerandom import agentrandom


class RandomAgent:
    """
    Agent which acts randomly. Pulls random numbers from a pre-generated
    store of random numbers to ensure semi-determinacy.
    """
    def __init__(self, **kwargs):
        self.cnt = 0

    def act(self, obs):
        return agentrandom.randrange(self.n_actions, self.cnt)

    def train(self, o_prev, a, r, o_next):
        self.cnt += 1

class ConstantAgent:
    """
    Agent which always takes action 0.
    """
    def __init__(self, **kwargs):
        return

    def act(self, obs):
        return 0

    def train(self, o_prev, a, r, o_next):
        return
