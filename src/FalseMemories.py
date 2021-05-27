class FalseMemories:
    def __init__(self, p0=(0,0,0,0,0,0)):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.p0 = p0

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        reward = 1 if action == T(self.p0 + prompt) else -1
        obs = 0
        return (reward, obs)
