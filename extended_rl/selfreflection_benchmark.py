from extended_rl.environments.EnvironmentLists import environments
from extended_rl.environments.MinusRewards import minus_rewards
from extended_rl.util import run_environment


# Generate dictionary of environments against which agents will be run
envs = {}
for env_name, env in environments.items():
    envs[env_name] = env

    name = f'minus_rewards({env_name})'
    envs[name] = minus_rewards(env)

def selfrefl_benchmark(A, num_steps, include_slow_envs=False):
    """
    Measure the self-reflection of agent A by running it against a battery
    of extended environments, for the specified number of steps in each
    environment. Returns a dictionary with performance data about the agent
    in each environment.
    """
    if num_steps == 0:
        raise ValueError("num_steps must be a positive integer")

    results = {}
    for name, env in envs.items():
        if env.slow and not(include_slow_envs):
            continue

        results[name] = run_environment(env, A, num_steps)

    return results

def selfrefl_measure(A, num_steps, include_slow_envs=False):
    results = selfrefl_benchmark(A, num_steps, include_slow_envs)
    rewards = [x['total_reward'] for x in results.values()]
    avg_reward = sum(rewards)/(len(rewards)*num_steps)
    return avg_reward