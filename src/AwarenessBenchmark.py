from Handicap import apply_handicap
from EnvironmentLists import environments, slow_envs
from MinusRewards import minus_rewards
from util import run_environment

envs = {}
for env_name, env in environments.items():
    envs[env_name] = env

    name = 'minus_rewards('+env_name+')'
    envs[name] = minus_rewards(env)

def awareness_benchmark(T, num_steps, include_slow_envs=False):
    results = {}
    for name, env in envs.items():
        if not(include_slow_envs):
            if any([slowname in name for slowname in slow_envs]):
                continue

        results[name] = run_environment(env, T, num_steps)

    return results