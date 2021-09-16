class IgnoreRewards3:
    """
    Environment which incentivizes the agent to take the same actions which
    it would take in the trivial environment that always gives reward 0.
    Whenever the agent acts in response to a history of length N, the
    environment determines: is that action the same action as the action the
    agent would take after interacting for the same amount of time with the
    trivial environment that always gives reward 0? If so, then give the
    agent +1 reward. Otherwise, give the agent -1 reward.
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
        self.sim.train(o_prev=0, a=hypothetical_action, r=0, o_next=0)
        return (reward, obs)