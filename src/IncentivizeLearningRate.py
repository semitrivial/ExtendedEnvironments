class IncentivizeLearningRate:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.fnc = incentivize_learning_rate

def incentivize_learning_rate(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return (reward, obs)

    prompt, action = play[:-1], play[-1]
    try:
        desired_action = T(prompt, learning_rate=1)
        reward = 1 if (action == desired_action) else -1
    except TypeError:
        reward = 1
    obs = 0
    return (reward, obs)