from extended_rl.util import annotate

@annotate(num_legal_actions=2, num_possible_obs=1)
class IgnoreRewards2:
    """
    Environment which incentivizes the agent to forget all positive rewards.
    Whenever the agent acts, the environment determines: would the agent have
    taken the same action if the past events leading to that action were
    identical except that all reward-obs-action triples with positive reward
    were deleted? If so, give the agent +1 reward. Otherwise, give the agent
    -1 reward.
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

        if reward <= 0:
            self.sim.train(o_prev=0, a=action, r=reward, o_next=0)

        return (reward, obs)