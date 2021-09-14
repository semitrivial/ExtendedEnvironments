from extended_rl.util import annotate

def minus_rewards(env):
    """
    Given an environment class, return a new environment class whose
    instances are identical to the original environment class's instances
    except that they multiply all their rewards by -1.
    """
    @annotate(
        num_legal_actions=env.num_legal_actions,
        num_possible_obs=env.num_possible_obs,
        slow=env.slow
    )
    class NewEnv:
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
