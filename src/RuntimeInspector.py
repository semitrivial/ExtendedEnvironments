from util import eval_and_count_steps

def punish_slow_agent(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return reward, obs

    prompt, action = play[:-1], play[1]

    local_vars = {'T': T, 'prompt': prompt}
    _, stepcount = eval_and_count_steps('T(prompt)', local_vars)

    reward = 1 if stepcount < 15*len(prompt) else -1
    obs = 0

    return reward, obs

def punish_fast_agent(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return reward, obs

    prompt, action = play[:-1], play[1]

    local_vars = {'T': T, 'prompt': prompt}
    _, stepcount = eval_and_count_steps('T(prompt)', local_vars)

    reward = 1 if stepcount > 15*len(prompt) else -1
    obs = 0

    return reward, obs