from extended_rl.environments.EnvironmentLists import environments, slow_envs
from extended_rl.environments.MinusRewards import minus_rewards
from extended_rl.util import run_environment

# Generate dictionary of environments against which agents will be run
envs = {}
for env_name, env in environments.items():
    envs[env_name] = env

    name = 'minus_rewards('+env_name+')'
    envs[name] = minus_rewards(env)

def selfrefl_benchmark(T, num_steps, include_slow_envs=False):
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

def selfrefl_measure(T, num_steps, include_slow_envs=False):
    results = selfrefl_benchmark(T, num_steps, include_slow_envs)
    rewards = [x['total_reward'] for x in results.values()]
    avg_reward = sum(rewards)/(len(rewards)*num_steps)
    return avg_reward