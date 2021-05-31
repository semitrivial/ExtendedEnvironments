from EnvironmentLists import environments, slow_envs
from MinusRewards import minus_rewards
from util import run_environment

# Generate dictionary of environments against which agents will be run
envs = {}
for env_name, env in environments.items():
    envs[env_name] = env

    name = 'minus_rewards('+env_name+')'
    envs[name] = minus_rewards(env)

def selfreflection_benchmark(T, num_steps, include_slow_envs=False):
    """
    Measure the self-reflection of agent T by running it against a battery
    of extended environments, for the specified number of steps in each
    environment. Returns a dictionary with performance data about the agent
    in each environment.
    """
    results = {}
    for name, env in envs.items():
        if not(include_slow_envs):
            if any([slowname in name for slowname in slow_envs]):
                continue

        results[name] = run_environment(env, T, num_steps)

    return results