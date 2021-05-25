class PunishDeterministicAgent:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return reward, obs

        prompt, action = play[:-1], play[-1]
        recomputed_action = T(prompt)
        reward = 1 if (action != recomputed_action) else -1
        obs = 0
        return reward, obs

class PunishNondeterministicAgent:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return reward, obs

        prompt, action = play[:-1], play[-1]
        recomputed_action = T(prompt)
        reward = 1 if (action == recomputed_action) else -1
        obs = 0
        return reward, obs
