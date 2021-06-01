class FalseMemories:
    """
    Environment which is intended to incentivize the agent to subjectively
    experience false memories. When the agent acts, the environment checks
    whether the agent would perform the same action if the true history
    preceding the action were preceded by a fixed false history. If so, the
    agent receives +1 reward. If not, the agent receives -1 reward. When
    initializing this environment, you can specify the false prehistory
    p0 (default is reward,obs,action,reward,obs,action=0,0,0,0,0,0).
    """
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
