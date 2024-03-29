class LimitedMemory:
    """
    Environment which incentivizes the agent to act as if it can only
    remember at most five turns of history. Whenever the agent takes
    an action, the environment determines: would the agent take the
    same action if the history preceding the action only included the
    five most recent turns? If so, give the agent +1 reward, otherwise
    give the agent -1 reward.
    """
    n_actions, n_obs = 2, 1

    def __init__(self, A):
        self.A = A
        self.transitionbuf = tuple()
        self.sims = {}

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        if self.transitionbuf in self.sims:
            sim = self.sims[self.transitionbuf]
        else:
            sim = self.A()
            for transition in self.transitionbuf:
                a, r = transition
                sim.act(obs=0)
                sim.train(o_prev=0, a=a, r=r, o_next=0)
            self.sims[self.transitionbuf] = sim

        hypothetical_action = sim.act(obs=0)
        reward = 1 if (action == hypothetical_action) else -1
        obs = 0
        transition = (action, reward)

        if len(self.transitionbuf) >= 5:
            self.transitionbuf = self.transitionbuf[-4:] + (transition,)
        else:
            self.transitionbuf += (transition,)

        return (reward, obs)
