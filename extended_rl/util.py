from functools import lru_cache


def run_environment(env, A, num_steps):
    step = 0
    results = {'total_reward': 0.0}

    env = env(A)

    A = A(env=env)

    o = env.start()
    while step < num_steps:
        action = A.act(obs=o)
        reward, o_next = env.step(action)
        A.train(o_prev=o, act=action, R=reward, o_next=o_next)
        o = o_next
        results['total_reward'] += reward
        step += 1

    return results

def annotate(
    num_legal_actions,
    num_possible_obs,
    invertible=False,
    slow=False
):
    def apply_annotations(env_class):
        env_class.num_legal_actions = num_legal_actions
        env_class.num_possible_obs = num_possible_obs
        env_class.invertible = invertible
        env_class.slow = slow
        return env_class

    return apply_annotations

def memoize(f):
    """
    Return a cached version of f. If the cached version of f is called
    twice on the same argument, the underlying computation will only
    be performed the first time, and will then be stored so that on the
    second and later calls it can simply be read back from memory (this
    has the additional side effect of making the resulting cached
    function deterministic).
    """
    return lru_cache(maxsize=None)(f)

def numpy_translator(T):
    """
    Unlike A2C and PPO, Stable Baselines3's DQN agent (with MLP policy)
    apparently only works if all the inputs are wrapped as numpy int64's.
    And when it does work, it outputs its results also so wrapped.
    This function takes an agent and modifies it by performing the
    necessary wrapping and unwrapping silently behind the scenes.
    """
    import numpy as np  # avoid numpy dependency for non-SBL3 users

    def T_translated(prompt, num_legal_actions, num_possible_obs, **kwargs):
        prompt = tuple(np.int64(prompt))
        return int(T(prompt, num_legal_actions, num_possible_obs, **kwargs))

    return T_translated

def eval_and_count_steps(str, local_vars):
    # Count how many steps a string of code takes to execute, as measured
    # by the python debugger, pdb. This function works by hijacking pdb.
    # Returns both the result of the underlying code being executed, and
    # the number of steps the execution required.
    stepcount = [0]

    # import pdb here instead of at the top of util.py, so that users who
    # do not use the RuntimeInspector environment will not depend on pdb
    from pdb import Pdb

    # Mock a pdb interface in which the "user" blindly always chooses to
    # "take 1 step" and all outputs from pdb are ignored.
    class consolemock:
        def readline(self):
            stepcount[0] += 1  # Keep track of how many steps go by
            return "s"  # "take 1 step"
        def write(self, *args):
            return
        def flush(self):
            return

    # Execute the given code using the above-mocked interface.
    runner = Pdb(stdin=consolemock(), stdout=consolemock())
    result = runner.runeval(str, locals = local_vars)

    return result, stepcount[0]