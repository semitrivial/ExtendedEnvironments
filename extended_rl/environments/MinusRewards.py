def minus_rewards(env):
    """
    Given an environment class, return a new environment class whose
    instances are identical to the original environment class's instances
    except that they multiply all their rewards by -1.
    """
    class NewEnv:
        n_actions, n_obs = env.n_actions, env.n_obs
        if hasattr(env, 'slow'):
            slow = env.slow

        def __init__(self, A, **kwargs):
            self.underlying = env(A, **kwargs)

        def start(self):
            return self.underlying.start()

        def step(self, action):
            reward, obs = self.underlying.step(action)
            return (-reward, obs)

    NewEnv.__name__ = f'minus_rewards({env.__name__})'
    NewEnv.__qualname__ = f'minus_rewards({env.__qualname__})'
    return NewEnv
