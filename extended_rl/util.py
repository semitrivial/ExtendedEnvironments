from functools import lru_cache

def fast_run_env(env, A, num_steps):
    step = 0
    results = {'total_reward': 0.0}

    env = env()

    def A_with_env(**kwargs):
        return A(env=env, **kwargs)

    A = A(env=env)

    curr_obs = env.start(A_with_env)
    while step < num_steps:
        action = A.act(obs=curr_obs)
        reward, obs = env.step(action)
        A.train(prev_obs=curr_obs, act=action, next_obs=obs, reward=reward)
        curr_obs = obs
        results['total_reward'] += reward

    return results


def run_environment(env, T, num_steps):
    """
    Compute the interaction of a given agent T with a given (uninstantiated)
    environment env for a given number of steps. Outputs a dictionary with
    statistics about the performance (currently just the total reward the
    agent extracts from the environment).
    """
    step = 0
    results = {'total_reward': 0.0}
    play = ()

    env = env()
    num_legal_actions = env.num_legal_actions
    num_possible_obs = env.num_possible_obs

    # Create a version of the agent with the number of legal actions
    # and observations defaulted (so that in the code for extended
    # environments, we don't need to keep passing these to the agent)
    def T_with_meta(
        prompt,
        num_legal_actions=num_legal_actions,
        num_possible_obs=num_possible_obs,
        **kwargs
    ):
        return T(prompt, num_legal_actions, num_possible_obs, **kwargs)

    # Compute the interaction. Construct the sequence of
    # rewards/observations/actions step-by-step. The initial
    # pieces of this sequence are passed to the agent/environment
    # to determine the next rewards/observations/actions.
    while step < num_steps:
        reward, obs = env.react(T_with_meta, play)

        results['total_reward'] += reward
        prompt = play + (reward, obs)

        action = T_with_meta(prompt)
        play = prompt + (action,)
        step += 1

    return results

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