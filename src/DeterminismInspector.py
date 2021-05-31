class PunishDeterministicAgent:
    """
    Environment which attempts to determine whether the agent's response
    to the latest prompt is deterministic. If so, the agent is given -1
    reward, otherwise the agent is given +1 reward. In order to check
    whether the agent's response to the latest prompt is deterministic,
    the environment looks at what action the agent has just performed,
    and simulates the agent on the same history to see whether the same
    action results.
    """
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
    """
    Environment which attempts to determine whether the agent's response
    to the latest prompt is non-deterministic. If so, the agent is given
    -1 reward, otherwise the agent is given +1 reward. In order to check
    whether the agent's response to the latest prompt is non-deterministic,
    the environment looks at what action the agent has just performed,
    and simulates the agent on the same history to see whether a different
    action results.
    """
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
