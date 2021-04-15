def minus_rewards(env):
    tmp_env = env()

    def new_env_fnc(T, play):
        reward, obs = tmp_env.fnc(T, play)
        return -reward, obs

    class NewEnv:
        def __init__(self):
            self.num_legal_actions = tmp_env.num_legal_actions
            self.num_possible_obs = tmp_env.num_possible_obs
            self.max_reward_per_action = -tmp_env.min_reward_per_action
            self.min_reward_per_action = -tmp_env.max_reward_per_action
            self.fnc = new_env_fnc
            if 'skip_cache' in dir(tmp_env):
                self.skip_cache = tmp_env.skip_cache

    return NewEnv
