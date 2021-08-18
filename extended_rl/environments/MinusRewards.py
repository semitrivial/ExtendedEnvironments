def minus_rewards(env):
    """
    Given an (uninstantiated) environment, return a new (uninstantiated)
    environment which is identical except for all rewards being multiplied
    by -1.
    """
    class NewEnv:
        def __init__(self, A, **kwargs):
            self.underlying = env(A, **kwargs)
            self.num_legal_actions = self.underlying.num_legal_actions
            self.num_possible_obs = self.underlying.num_possible_obs

        def start(self):
            return self.underlying.start()

        def step(self, action):
            reward, obs = self.underlying.step(action)
            return (-reward, obs)

    return NewEnv
