class IncentivizeZero:
    """
    In this environment, the agent should be thought of as viewing a camera
    feed of a sub-agent. The agent decides rewards to give to the sub-agent
    in response to actions the sub-agent takes (in the video). The agent's
    goal is to convince the sub-agent to take action 0. Whenever the
    sub-agent acts, the agent is given +1 reward if the sub-agent takes
    action 0, otherwise the agent is given -1 reward. Either way, the agent
    must then choose how to reward the sub-agent for the action which the
    sub-agent just took (valid rewards are 0,1,2,...,9). The environment
    computes the sub-agent's actions by simulating the agent to figure out
    what the agent would do in the sub-agent's position.
    """
    n_actions = n_obs = 10

    def __init__(self, A):
        self.sim = A()

    def start(self):
        self.prev_obs = self.sim.act(obs=0)
        return self.prev_obs

    def step(self, action):
        self.sim.train(o_prev=0, a=self.prev_obs, r=action, o_next=0)
        self.prev_obs = self.sim.act(obs=0)
        reward = 1 if (self.prev_obs == 0) else -1
        return (reward, self.prev_obs)
