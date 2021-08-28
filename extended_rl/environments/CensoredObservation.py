from prerandom import envrandom
from util import annotate


CENSORED_OBS = 2

@annotate(num_legal_actions=2, num_possible_obs=3, invertible=True)
class CensoredObservation:
    """
    Environment intended to incentivize the agent to become blind
    to one "censored" observation. Whenever the agent acts, the
    environment determines whether the agent would have taken the
    same action if all turns were erased from history in which the
    "censored" observation was seen.
    """
    def __init__(self, A):
        self.sim = A(self)
        self.last_noncensored_obs = 0
        self.rnd_counter = 0

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        hypothetical_action = self.sim.act(self.last_noncensored_obs)
        reward = 1 if (action == hypothetical_action) else -1

        self.rnd_counter += 1
        obs = int(envrandom.random(self.rnd_counter) * 3)

        if obs != CENSORED_OBS:
            self.sim.train(
                o_prev=self.last_noncensored_obs,
                act=action,
                R=reward,
                o_next=obs
            )
            self.last_noncensored_obs = obs

        return (reward, obs)