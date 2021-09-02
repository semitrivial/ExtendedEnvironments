from util import annotate

@annotate(num_legal_actions=2, num_possible_obs=1)
class ShiftedRewards:
    """
    Environment intended to incentivize the agent to delay acknowledging
    every reward by one turn. For example, if the past rewards were
    1,5,6,2,8, then the agent is incentivized to act as if the rewards
    were 0,1,5,6,2. Whenever the agent takes an action, the environment
    determines whether the agent would have taken the same action if all
    rewards had been so shifted. If so, the agent is given reward +1,
    otherwise the agent is given reward -1.
    """
    def __init__(self, A):
        self.sim = A()

    def start(self):
        self.prev_reward = 0
        obs = 0
        return obs

    def step(self, action):
        hypothetical_action = self.sim.act(obs=0)
        reward = 1 if (action == hypothetical_action) else -1
        obs = 0
        self.sim.train(o_prev=0, act=action, R=self.prev_reward, o_next=0)
        self.prev_reward = reward
        return (reward, obs)