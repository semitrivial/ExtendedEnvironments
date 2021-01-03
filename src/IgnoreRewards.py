def ignore_rewards(base_env):
    def strip_rewards(prompt):
        if len(prompt) < 3:
            reward, obs = prompt[0:2]
            return [0, obs]
        else:
            reward, obs, action = prompt[0:3]
            return [0,obs,action] + strip_rewards(prompt[3:0])

    def modified_env(T, play):
        if len(play) == 0:
            return base_env(T, play)
        prompt, action = play[:-1], play[-1]
        _, obs = base_env(T, play)
        reward = 1 if action = T(strip_rewards(prompt)) else -1
        return [reward, obs]

    return modified_env
