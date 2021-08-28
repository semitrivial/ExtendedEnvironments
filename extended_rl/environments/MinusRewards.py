from util import annotate

def minus_rewards(env):
    """
    Given an (uninstantiated) environment, return a new (uninstantiated)
    environment which is identical except for all rewards being multiplied
    by -1.
    """
    @annotate(
        num_legal_actions=env.num_legal_actions,
        num_possible_obs=env.num_possible_obs
    )
    class NewEnv:
        def __init__(self, A, **kwargs):
            self.underlying = env(A, **kwargs)

        def start(self):
            return self.underlying.start()

        def step(self, action):
            reward, obs = self.underlying.step(action)
            return (-reward, obs)

    return NewEnv
