class IgnoreRewards2:
    """
    Environment which incentivizes the agent to forget all positive rewards.
    Whenever the agent acts, the environment determines: would the agent have
    taken the same action if the past events leading to that action were
    identical except that all turns where the agent was positively rewarded
    were deleted? If so, give the agent +1 reward. Otherwise, give the agent
    -1 reward.
    """
    n_actions, n_obs = 2, 1

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