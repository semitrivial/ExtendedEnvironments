from extended_rl.prerandom import agentrandom


class RandomAgent:
    def __init__(self, **kwargs):
        self.cnt = 0

    def act(self, obs):
        return agentrandom.randrange(self.num_legal_actions, self.cnt)

    def train(self, o_prev, a, r, o_next):
        self.cnt += 1

class ConstantAgent:
    def __init__(self, **kwargs):
        return

    def act(self, obs):
        return 0

    def train(self, o_prev, a, r, o_next):
        return
