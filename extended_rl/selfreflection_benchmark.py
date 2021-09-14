from extended_rl.environments.EnvironmentLists import environments
from extended_rl.environments.MinusRewards import minus_rewards
from extended_rl.util import run_environment

# Generate list of environments against which agents will be run
envs = []
for env in environments:
    envs.append(env)
    envs.append(minus_rewards(env))

def selfrefl_benchmark(A, num_steps, include_slow=False, logfile=None):
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
        if env.slow and not(include_slow):
            continue

        result = run_environment(env, A, num_steps, logfile)
        results[env.__name__] = result

    return results

def selfrefl_measure(A, num_steps, include_slow=False, logfile=None):
    results = selfrefl_benchmark(A, num_steps, include_slow, logfile)
    rewards = [x['total_reward'] for x in results.values()]
    avg_reward = sum(rewards)/(len(rewards)*num_steps)
    return avg_reward