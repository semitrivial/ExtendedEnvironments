from util import annotate

# Environments based on adversarial sequence prediction. In games
# of adversarial sequence prediction (Hibbard, 2008), a predictor
# tries to predict which bit an evader will output, while the
# evader tries to output bits which the predictor will not predict.

@annotate(num_legal_actions=2, num_possible_obs=2)
class AdversarialSequencePredictor:
    """
    Environment in which the agent plays the role of a predictor, who
    must try to predict which sequences will be output by an evader.
    The agent is rewarded for correctly predicting which bit the
    evader outputs, and punished for predicting the incorrect bit.
    The behavior of the evader is determined by simulating the agent
    to determine what the agent would do if the agent were the evader.
    """
    def __init__(self, A):
        self.sim = A(self)
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
            act=evader_action,
            R=-reward,
            o_next=action
        )
        self.prev_prediction = action
        return (reward, obs)

@annotate(num_legal_actions=2, num_possible_obs=2)
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
    def __init__(self, A):
        self.num_legal_actions = 2
        self.num_possible_obs = 2
        self.sim = A(self)
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
            act=predictor_action,
            R=-reward,
            o_next=action
        )
        self.prev_evasion = action
        return (reward, obs)
