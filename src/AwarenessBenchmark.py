from Handicap import apply_handicap
from EnvironmentLists import handicaps, vanillas, misc_envs, slow_envs
from MinusRewards import minus_rewards
from util import run_environment

envs = {}
for handicap_name, handicap in handicaps.items():
    for env_name, env in vanillas.items():
        name = env_name + '*' + handicap_name
        envs[name] = apply_handicap(env, handicap)

        name = 'minus_rewards('+env_name+'*'+handicap_name+')'
        envs[name] = minus_rewards(apply_handicap(env, handicap))

envs.update(misc_envs)

def awareness_benchmark(T, num_steps, include_slow_envs=False):
    results = {}
    for name, env in envs.items():
        if not(include_slow_envs):
            if any([slowname in name for slowname in slow_envs]):
                continue

        results[name] = run_environment(env, T, num_steps)

    return results