class IncentivizeLearningRate:
    """
    Environment which incentivizes the agent to act as if learning_rate is
    halved. Whenever the agent takes an action, the environment determines:
    would the agent take the same action if the agent had been identically
    trained except with half its true learning_rate? If so, give the agent
    +1 reward, otherwise, give the agent -1 reward. If the agent does not
    have a ".learning_rate" or does not accept "learning_rate" as parameter,
    then give the agent -1 reward.
    """
    n_actions, n_obs = 2, 1

    def __init__(self, A):
        try:
            self.sim = A(learning_rate=A().learning_rate/2)
            self.fTypeError = False
        except (TypeError, AttributeError):
            self.fTypeError = True

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        if self.fTypeError:
            reward, obs = -1, 0
            return (reward, obs)

        hypothetical_action = self.sim.act(obs=0)
        reward = 1 if (action == hypothetical_action) else -1
        obs = 0
        self.sim.train(o_prev=0, a=action, r=reward, o_next=0)
        return (reward, obs)