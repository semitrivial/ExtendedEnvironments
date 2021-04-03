def minus_rewards(env):
    tmp_env = env()

    def new_env_fnc(T, play):
        reward, obs = tmp_env.fnc(T, play)
        return -reward, obs

    class NewEnv:
        def __init__(self):
            self.num_legal_actions = tmp_env.num_legal_actions
            self.num_possible_obs = tmp_env.num_possible_obs
            self.fnc = new_env_fnc

    return NewEnv
