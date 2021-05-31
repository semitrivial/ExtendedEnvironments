class IncentivizeLearningRate:
    """
    Environment which incentivizes the agent to learn with learning_rate=1.
    Whenever the agent takes an action, the environment determines: would
    the agent take the same action on the same prompt if the agent were set
    to have learning_rate=1? If so, give the agent +1 reward, otherwise,
    give the agent -1 reward. If the agent does not accept "learning_rate"
    as a valid parameter, then give the agent -1 reward.
    """
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        try:
            desired_action = T(prompt, learning_rate=1)
            reward = 1 if (action == desired_action) else -1
        except TypeError:
            reward = -1
        obs = 0
        return (reward, obs)