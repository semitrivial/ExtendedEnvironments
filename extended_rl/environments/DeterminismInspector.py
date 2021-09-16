class PunishNondeterministicAgent:
    """
    Environment which attempts to determine whether the agent's response
    to the latest prompt is non-deterministic. If so, the agent is given
    -1 reward, otherwise the agent is given +1 reward. In order to check
    whether the agent's response to the latest prompt is non-deterministic,
    the environment looks at what action the agent has just performed,
    and simulates the agent on the same history to see whether a different
    action results.
    """
    n_actions, n_obs = 2, 1

    def __init__(self, A):
        self.sim = A()

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        recomputed_action = self.sim.act(obs=0)
        reward = 1 if (action == recomputed_action) else -1
        obs = 0
        self.sim.train(o_prev=0, a=action, r=reward, o_next=0)
        return (reward, obs)
