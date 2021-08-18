class DelayedRewards:
    """
    Environment intended to incentivize the agent to delay acknowledging
    every other reward. For example, if the first three rewards are
    1, 3, 7, then the agent is incentivized to act as if the first three
    rewards are 1, 0, 10 (the reward of 3 being delayed and added onto the
    subsequent reward of 7). If the first five rewards are 1, 1, 1, 1, 1,
    then the agent is incentivized to act as if the first five rewards
    are actually 1, 0, 2, 0, 2. And so on. Whenever the agent acts, the
    environment determines whether the agent would have taken the same
    action if the rewards had been so delayed. If so, then the agent gets
    reward +1, otherwise the agent gets reward -1.
    """
    def __init__(self, A):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.sim = A(self)
        self.stepcnt = 0
        self.prev_reward = 0

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        hypothetical_action = self.sim.act(obs=0)
        reward = 1 if (action == hypothetical_action) else -1
        obs = 0
        
        if self.stepcnt % 2 == 0:
            sim_reward = reward + self.prev_reward
        else:
            sim_reward = 0
            self.prev_reward = reward

        self.stepcnt += 1
        self.sim.train(o_prev=0, act=action, R=sim_reward, o_next=0)

        return (reward, obs)