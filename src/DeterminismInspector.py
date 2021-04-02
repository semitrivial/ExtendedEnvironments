def punish_deterministic_agent(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return reward, obs

    prompt, action = play[:-1], play[-1]
    recomputed_action = T(prompt)
    reward = 1 if (action != recomputed_action) else -1
    obs = 0
    return reward, obs

punish_deterministic_agent.skip_cache = True

def punish_nondeterministic_agent(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return reward, obs

    prompt, action = play[:-1], play[-1]
    recomputed_action = T(prompt)
    reward = 1 if (action == recomputed_action) else -1
    obs = 0
    return reward, obs

punish_nondeterministic_agent.skip_cache = True