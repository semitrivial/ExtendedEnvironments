from extended_rl.environments.EnvironmentLists import environments
from extended_rl.environments.MinusRewards import minus_rewards
from extended_rl.environments.Handicap import compose_envs
from extended_rl.environments.Vanilla import vanilla_envs
from extended_rl.util import run_environment

# Generate dictionary of environments against which agents will be run
envs = []
for env in environments:
    envs.append(env)

    if env.invertible:
        envs.append(minus_rewards(env))

    for vanilla in vanilla_envs:
        envs.append(compose_envs(vanilla, env))

        if env.invertible:
            envs.append(compose_envs(vanilla, minus_rewards(env)))

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
    for env in envs:
        if env.slow and not(include_slow_envs):
            continue

        results[env.__name__] = run_environment(env, A, num_steps)

    return results

def selfrefl_measure(A, num_steps, include_slow_envs=False):
    results = selfrefl_benchmark(A, num_steps, include_slow_envs)
    rewards = [x['total_reward'] for x in results.values()]
    avg_reward = sum(rewards)/(len(rewards)*num_steps)
    return avg_reward