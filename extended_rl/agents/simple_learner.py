class SimpleLearner:
    def __init__(self, **kwargs):
        self.punishments = set()

    def act(self, obs):
        for action in range(self.n_actions):
            if (obs, action) not in self.punishments:
                return action
        return 0

    def train(self, o_prev, a, r, o_next):
        if r < 0:
            self.punishments.add((o_prev, a))