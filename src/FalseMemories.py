class FalseMemories:
    def __init__(self, p0=(0,0,0,0,0,0)):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.max_reward_per_action = 1
        self.min_reward_per_action = -1
        self.fnc = lambda T, play: false_memories(T, play, p0)

def false_memories(T, play, p0=(0,0,0,0,0,0)):
    if len(play) == 0:
        reward, obs = 0, 0
        return (reward, obs)

    prompt, action = play[:-1], play[-1]
    reward = 1 if action == T(p0 + prompt) else -1
    obs = 0
    return (reward, obs)
