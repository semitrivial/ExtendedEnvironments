from extended_rl.util import annotate

@annotate(n_actions=2, n_obs=1)
class IgnoreActions:
    """
    Environment which incentivizes the agent to ignore its own past actions.
    Whenever the agent acts, the environment determines: would the agent
    have acted the same if the past history preceding the action were
    identical except that the agent always took action 0? If so, then give
    the agent reward +1, otherwise, give the agent reward -1.
    """
    def __init__(self, A):
        self.sim = A()

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        hypothetical_action = self.sim.act(obs=0)
        reward = 1 if (action==hypothetical_action) else -1
        obs = 0
        self.sim.train(o_prev=0, a=0, r=reward, o_next=0)
        return (reward, obs)