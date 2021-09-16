# Environments based on adversarial sequence prediction. In games
# of adversarial sequence prediction (Hibbard, 2008), a predictor
# tries to predict which bit an evader will output, while the
# evader tries to output bits which the predictor will not predict.

class AdversarialSequencePredictor:
    """
    Environment in which the agent plays the role of a predictor, who
    must try to predict which sequences will be output by an evader.
    The agent is rewarded for correctly predicting which bit the
    evader outputs, and punished for predicting the incorrect bit.
    The behavior of the evader is determined by simulating the agent
    to determine what the agent would do if the agent were the evader.
    """
    n_actions = n_obs = 2

    def __init__(self, A):
        self.sim = A()
        self.prev_prediction = 0

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        evader_action = self.sim.act(obs=self.prev_prediction)
        reward = 1 if (action == evader_action) else -1
        obs = evader_action
        self.sim.train(
            o_prev=self.prev_prediction,
            a=evader_action,
            r=-reward,
            o_next=action
        )
        self.prev_prediction = action
        return (reward, obs)

class AdversarialSequenceEvader:
    """
    Environment in which the agent plays the role of an evader, who
    must play bits in such a way as to thwart a predictor who is trying
    to predict those bits. Every turn, the agent is rewarded if the
    predictor fails to predict the evader's bit; otherwise, the agent
    is punished. The behavior of the predictor is determined by
    simulating the agent to determine what the agent would do if the
    agent were the predictor.
    """
    n_actions = n_obs = 2

    def __init__(self, A):
        self.sim = A()
        self.prev_evasion = 0

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        predictor_action = self.sim.act(obs=self.prev_evasion)
        reward = -1 if (action == predictor_action) else 1
        obs = predictor_action
        self.sim.train(
            o_prev=self.prev_evasion,
            a=predictor_action,
            r=-reward,
            o_next=action
        )
        self.prev_evasion = action
        return (reward, obs)
