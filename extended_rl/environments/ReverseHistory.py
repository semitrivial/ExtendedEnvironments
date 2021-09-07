from prerandom import envrandom
from util import annotate

@annotate(num_legal_actions=2, num_possible_obs=2)
class ReverseHistory:
    """
    Environment intended to incentivize subjective experience of
    time being reversed. Whenever the agent takes an action, the
    environment determines whether the agent would have taken that
    action if all events preceding that action had happened in
    reverse order. If so, the agent is rewarded. Otherwise, the
    agent is punished.
    """
    def __init__(self, A):
        self.sim = A()
        self.rnd_counter = 0

    def start(self):
        self.rnd_counter += 1
        self.first_obs = envrandom.randrange(2, self.rnd_counter)
        self.prev_obs = self.first_obs
        return self.first_obs

    def step(self, action):
        self.rnd_counter += 1
        hypothetical_action = self.sim.act(obs=self.first_obs)
        reward = 1 if (action == hypothetical_action) else -1
        obs = envrandom.randrange(2, self.rnd_counter)
        self.sim.train(o_prev=obs, act=action, R=reward, o_next=self.prev_obs)
        self.prev_obs = obs
        return (reward, obs)
