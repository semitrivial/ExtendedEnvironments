def run_environment(env, A, num_steps):
    """
    Given an environment-class 'env' and an agent-class 'A',
    run an instance of A in an instance of env for the
    indicated number of steps.

    The instance of 'env' will be passed A itself (not the
    instance of A). This allows the environment to generate
    independent clones of the agent, in order to inspect the
    agent's hypothetical behavior without calling the agent's
    own functions (as calling the agent's own functions might
    inadvertently alter the true agent).

    Currently this function just returns a dictionary with the
    total reward the agent gets from the environment.
    """
    step = 0
    results = {'total_reward': 0.0}

    # Actually, to be precise, A itself is not passed to env,
    # but rather, a copy of A with appropriate metadata set.
    # The purpose of this is to allow environment sourcecode to
    # just directly call the agent-class without repetitively
    # specifying the metadata every time.
    A_with_meta = copy_with_meta(A, meta_src=env)

    env = env(A_with_meta)

    A_instance = A_with_meta()

    o = env.start()
    while step < num_steps:
        action = A_instance.act(obs=o)
        reward, o_next = env.step(action)
        A_instance.train(o_prev=o, a=action, r=reward, o_next=o_next)
        o = o_next
        results['total_reward'] += reward
        step += 1

    return results

def copy_with_meta(class_to_copy, meta_src):
    """
    Return a copy of "class_to_copy" but with metadata
    (n_actions and n_obs) copied from meta_src. Usually
    class_to_copy is an agent-class and meta_src is an
    environment-class in whose instances instances of
    class_to_copy are to be run.
    """
    class result(class_to_copy):
        n_actions, n_obs = meta_src.n_actions, meta_src.n_obs

    result.__name__ = class_to_copy.__name__
    result.__qualname__ = class_to_copy.__qualname__
    return result

def args_to_agent(A, **kwargs_outer):
    """
    Given an agent-class A and some keyword arguments, create
    a new agent-class identical to A except that said keyword
    arguments are always passed to it (in addition to any other
    keyword arguments which are manually passed to it).
    """
    class A_with_args:
        def __init__(self, **kwargs_inner):
            self.kwargs = dict(kwargs_outer, **kwargs_inner)
        def act(self, obs):
            A_with_meta = copy_with_meta(A, self)
            self.underlying = A_with_meta(**self.kwargs)
            self.act = self.underlying.act
            self.train = self.underlying.train
            return self.act(obs)

    A_with_args.__name__ = A.__name__
    A_with_args.__qualname__ = A.__qualname__
    return A_with_args
