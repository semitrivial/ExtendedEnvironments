from agents.Q import Q_learner
from util import annotate


class recurrent_Q:
    def __init__(self, n=9, epsilon=0.9, alpha=0.1, gamma=0.9):
        @annotate(
            num_legal_actions=self.num_legal_actions,
            num_possible_obs=self.num_possible_obs
        )
        class Q_learner_with_meta(Q_learner):
            pass

        self.underlying = Q_learner_with_meta(epsilon, alpha, gamma)
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
