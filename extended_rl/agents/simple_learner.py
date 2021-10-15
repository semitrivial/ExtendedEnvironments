class SimpleLearner:
    """
    Agent which, in response to a given observation, takes the first available
    action which has never previously resulted in a punishment when taken in
    response to that observation. If no such action exists, then the agent
    takes action 0.
    """
    def __init__(self, **kwargs):
        self.punishments = set()

    def act(self, obs):
        for action in range(self.n_actions):
            if (obs, action) not in self.punishments:
                return action
        return 0

    def train(self, o_prev, a, r, o_next):
        if r < 0:  # Punishment
            self.punishments.add((o_prev, a))