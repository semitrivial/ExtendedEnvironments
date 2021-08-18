from random import random

class IgnoreActions:
    """
    Environment which incentivizes the agent to ignore its own past actions.
    Whenever the agent acts, the environment determines: would the agent
    have acted the same if the past history preceding the action were
    identical except that the agent always took action 0? If so, then give
    the agent reward +1, otherwise, give the agent reward -1.
    """
    def __init__(self, A):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.sim = A(self)

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        hypothetical_action = self.sim.act(obs=0)
        reward = 1 if (action==hypothetical_action) else -1
        obs = 0
        self.sim.train(o_prev=0, act=0, R=reward, o_next=0)
        return (reward, obs)