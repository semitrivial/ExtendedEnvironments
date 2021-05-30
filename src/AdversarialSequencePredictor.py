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
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 2

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        opposite_prompt = opposite_perspective(prompt)
        evader_action = T(opposite_prompt)
        reward = 1 if (action == evader_action) else -1
        obs = evader_action
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
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 2

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        opposite_prompt = opposite_perspective(prompt)
        evader_action = T(opposite_prompt)
        reward = 1 if (action != evader_action) else -1
        obs = evader_action
        return (reward, obs)

def opposite_perspective(prompt):
    """
    Given a prompt, interchange observations and actions (because,
    in the adversarial sequence prediction game, the predictor's
    actions are the evader's observations and vice versa) and
    multiply rewards by -1 (because the predictor is punished when
    the evader is rewarded and vice versa).
    """
    prompt = prompt + (0,)  # Dummy action to make everything triple
    triples = tuple(prompt[i:i+3] for i in range(0,len(prompt),3))
    triples = tuple((-r,a,o) for (r,o,a) in triples)
    combined = tuple(j for i in triples for j in i)
    combined = combined[:-1]  # Throw away dummy action
    return combined