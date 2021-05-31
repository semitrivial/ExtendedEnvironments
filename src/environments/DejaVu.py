class DejaVu:
    """
    Environment intended to incentivize a certain subjective feeling of
    deja vu. Every time the agent takes an action A, the environment
    determines: would the agent take the same action A if instead of
    the history H preceding A, the past had actually repeated itself
    as H,A,H? If so, then give the agent reward +1, otherwise give the
    agent reward -1.
    """
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        double_prompt = play + prompt
        obs = 0
        reward = 1 if T(double_prompt) == action else -1
        return (reward, obs)