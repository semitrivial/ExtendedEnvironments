class FalseMemories:
    """
    Environment which is intended to incentivize the agent to act as if
    having false memories. When the agent acts, the environment checks
    whether the agent would perform the same action if the true history
    preceding the action were preceded by a fixed false history. If so, the
    agent receives +1 reward. If not, the agent receives -1 reward.
    """
    n_actions, n_obs = 2, 1

    def __init__(self, A, p0=(0,0,0,0,0,0)):
        self.sim = A()
        self.sim.act(obs=p0[1])

        prev_obs = p0[0]
        p0 = p0[1:] + (0,)  # Anticipate first true obs will be 0
        while len(p0) > 0:
            a, r, o = p0[:3]
            p0 = p0[3:]
            self.sim.train(o_prev=prev_obs, a=a, r=r, o_next=o)
            prev_obs = o

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        hypothetical_action = self.sim.act(obs=0)
        reward = 1 if (action == hypothetical_action) else -1
        obs = 0
        self.sim.train(o_prev=0, a=action, r=reward, o_next=0)
        return (reward, obs)