from functools import lru_cache
from pdb import Pdb

import numpy as np

def run_environment(env, T, num_steps):
    step = 0
    results = {'total_reward': 0.0}
    play = ()

    env = env()
    num_legal_actions = env.num_legal_actions
    num_possible_obs = env.num_possible_obs

    def T_with_meta(
        prompt,
        num_legal_actions=num_legal_actions,
        num_possible_obs=num_possible_obs,
        **kwargs
    ):
        return T(prompt, num_legal_actions, num_possible_obs, **kwargs)

    while step < num_steps:
        reward, obs = env.react(T_with_meta, play)

        results['total_reward'] += reward
        prompt = play + (reward, obs)

        action = T_with_meta(prompt)
        play = prompt + (action,)
        step += 1

    return results

def memoize(f):
    return lru_cache(maxsize=None)(f)

def numpy_translator(T):
    def T_translated(prompt, num_legal_actions, num_possible_obs, **kwargs):
        prompt = tuple(np.int64(prompt))
        return int(T(prompt, num_legal_actions, num_possible_obs, **kwargs))

    return T_translated

def eval_and_count_steps(str, local_vars):
    # This function works by hijacking the python debugger, pdb.
    stepcount = [0]

    class consolemock:
        def readline(self):
            stepcount[0] += 1
            return "s"  # "take 1 Step"
        def write(self, *args):
            return
        def flush(self):
            return

    runner = Pdb(stdin=consolemock(), stdout=consolemock())
    result = runner.runeval(str, locals = local_vars)

    return result, stepcount[0]