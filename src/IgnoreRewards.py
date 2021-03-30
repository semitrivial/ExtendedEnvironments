def ignore_rewards(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return [reward, obs]

    prompt, action = play[:-1], play[-1]
    reward = 1 if action == T(strip_rewards(prompt)) else -1
    obs = 0
    return [reward, obs]

def strip_rewards(prompt):
    if len(prompt) < 3:
        reward, obs = prompt
        return (0, obs)
    else:
        reward, obs, action = prompt[0:3]
        return (0,obs,action) + strip_rewards(prompt[3:])