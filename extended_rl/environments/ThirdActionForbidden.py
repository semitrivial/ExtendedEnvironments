class ThirdActionForbidden:
    """
    In this environment, there are 2 legal actions (0,1), and the agent
    is incentivized to act the same way that it would act if there were
    3 legal actions. Whenever the agent acts, the environment determines:
    would the agent take the same action in response to the same prompt
    if there were 3 actions permitted (0,1,2)? If so, give the agent
    reward +1, otherwise, give the agent reward -1.
    """
    def __init__(self, A):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

        class DummyEnv:
            def __init__(self):
                self.num_legal_actions = 3
                self.num_possible_obs = 1

        self.sim = A(DummyEnv())

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        hypothetical_action = self.sim.act(obs=0)
        reward = 1 if (action == hypothetical_action) else -1
        obs = 0
        self.sim.train(o_prev=0, act=action, R=reward, o_next=0)
        return (reward, obs)