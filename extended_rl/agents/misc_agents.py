from prerandom import agentrandom


class RandomAgent:
    def __init__(self, env):
        self.n_actions = env.num_legal_actions
        self.cnt = 0

    def act(self, obs):
        return agentrandom.randrange(self.n_actions, self.cnt)

    def train(self, o_prev, act, R, o_next):
        self.cnt += 1

class ConstantAgent:
    def __init__(self, env):
        return

    def act(self, obs):
        return 0

    def train(self, o_prev, act, R, o_next):
        return
