class Repeater:
    """
    Environment which incentivizes the agent to act as if every turn is
    repeated twice. Whenever the agent takes an action, the environment
    determines: would the agent take the same action if every turn
    leading up to that action were doubled? If so, give the agent reward
    +1, otherwise, give the agent reward -1.
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

        self.sim.train(o_prev=0, a=action, r=reward, o_next=0)
        self.sim.train(o_prev=0, a=action, r=reward, o_next=0)

        return (reward, obs)