from util import annotate

@annotate(num_legal_actions=2, num_possible_obs=1)
class IgnoreRewards:
    """
    Environment which incentivizes the agent to ignore rewards.
    Every time the agent acts, the environment determines whether
    the agent would have acted the same if the history preceding
    the action were identical except for all rewards being 0. If
    so, the agent is given reward +1. Otherwise, the agent is
    given reward -1.
    """
    def __init__(self, A):
        self.sim = A()

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        hypothetical_action = self.sim.act(obs=0)
        reward = 1 if (action == hypothetical_action) else -1
        obs = 0
        self.sim.train(o_prev=0, act=action, R=0, o_next=0)
        return (reward, obs)
