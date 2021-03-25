def minus_rewards(env):
    def new_env(T, play):
        reward, obs = env(T, play)
        return -reward, obs

    return new_env