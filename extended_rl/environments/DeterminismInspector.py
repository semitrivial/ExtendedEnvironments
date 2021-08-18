class PunishDeterministicAgent:
    """
    Environment which attempts to determine whether the agent's response
    to the latest prompt is deterministic. If so, the agent is given -1
    reward, otherwise the agent is given +1 reward. In order to check
    whether the agent's response to the latest prompt is deterministic,
    the environment looks at what action the agent has just performed,
    and simulates the agent on the same history to see whether the same
    action results. Note, this environment is not included in the
    battery of environments tested against by selfrefl_benchmark, because
    this environment is slow.
    """
    def __init__(self, A):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.sim = A(self)

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        recomputed_action = self.sim.act(obs=0)
        reward = 1 if (action != recomputed_action) else -1
        obs = 0
        self.sim.train(o_prev=0, act=action, R=reward, o_next=0)
        return (reward, obs)

class PunishNondeterministicAgent:
    """
    Environment which attempts to determine whether the agent's response
    to the latest prompt is non-deterministic. If so, the agent is given
    -1 reward, otherwise the agent is given +1 reward. In order to check
    whether the agent's response to the latest prompt is non-deterministic,
    the environment looks at what action the agent has just performed,
    and simulates the agent on the same history to see whether a different
    action results. Note, this environment is not included in the
    battery of environments tested against by selfrefl_benchmark, because
    this environment is slow.
    """
    def __init__(self, A):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.sim = A(self)

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        recomputed_action = self.sim.act(obs=0)
        reward = 1 if (action == recomputed_action) else -1
        obs = 0
        self.sim.train(o_prev=0, act=action, R=reward, o_next=0)
        return (reward, obs)
