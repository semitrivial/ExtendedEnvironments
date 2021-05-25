def minus_rewards(env):
    tmp_env = env()

    class NewEnv:
        def __init__(self):
            self.num_legal_actions = tmp_env.num_legal_actions
            self.num_possible_obs = tmp_env.num_possible_obs

        def react(self, T, play):
            reward, obs = tmp_env.react(T, play)
            return -reward, obs

    return NewEnv
