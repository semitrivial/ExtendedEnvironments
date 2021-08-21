from agents.Q import Q_learner

class recurrent_Q:
    def __init__(self, env, n=3, epsilon=0.9, alpha=0.1, gamma=0.9):
        self.underlying = Q_learner(env, epsilon, alpha, gamma)
        self.history = tuple()
        self.n = n

    def act(self, obs):
        return self.underlying.act(self.history[:-1] + (obs,))

    def train(self, o_prev, act, R, o_next):
        old_history = self.history
        self.history = (self.history + (act, R, o_next))[-self.n:]
        self.underlying.train(
            o_prev=old_history,
            act=act,
            R=R,
            o_next=self.history
        )
