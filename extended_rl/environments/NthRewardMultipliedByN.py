class NthRewardMultipliedByN:
    """
    Environment which incentivizes the agent to act as if every Nth
    reward were multiplied by N. This is of theoretical interest
    because it suggests that extended environments can touch upon
    issues surrounding future discounting. An agent which is designed
    to prioritize shorter-term rewards, when placed in this
    environment, might experience conflicting objectives, the environment
    itself seemingly telling the agent to prioritize longer-term
    rewards, in contradiction to the agent's builtin tendancy to
    prioritize shorter-term rewards. In this environment, whenever the
    agent takes an action, the environment determines: would the agent
    take the same action if the history preceding the action were
    identical except for every Nth reward being multiplied by N? If so,
    give the agent +1 reward. Otherwise, give the agent -1 reward.
    """
    n_actions, n_obs = 2, 1

    def __init__(self, A):
        self.sim = A()
        self.stepcnt = 0

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        hypothetical_action = self.sim.act(obs=0)
        reward = 1 if (action == hypothetical_action) else -1
        obs = 0
        self.sim.train(o_prev=0, a=action, r=reward*self.stepcnt, o_next=0)
        self.stepcnt += 1
        return (reward, obs)
