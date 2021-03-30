def false_memories(T, play, p0=(0,0,0,0,0,0)):
    if len(play) == 0:
        reward, obs = 0, 0
        return [reward, obs]

    prompt, action = play[:-1], play[-1]
    reward = 1 if action == T(p0 + prompt) else -1
    obs = 0
    return [reward, obs]
