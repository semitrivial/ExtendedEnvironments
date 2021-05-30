import sys
import random
from collections import deque

import numpy as np
import torch

from agents.misc_agents import random_agent, constant_agent
from agents.naive_learner import naive_learner
from agents.SBL3_agents import (
    agent_A2C, agent_DQN, agent_PPO,
    clear_cache_A2C, clear_cache_DQN, clear_cache_PPO
)
from agents.reality_check import reality_check
from AwarenessBenchmark import awareness_benchmark
from util import memoize

seed, n_steps = 0, 100

args = deque(sys.argv[1:])
while args:
    arg = args.popleft()
    if arg == 'seed':
        seed = int(args.popleft())
    elif arg == 'steps':
        n_steps = int(args.popleft())
    else:
        raise ValueError("Unrecognized commandline argument")

print("Testing agents with seed="+str(seed)+", n_steps="+str(n_steps))

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

def measure_agent(name, agent):
    print("Testing "+name+"...")
    result = awareness_benchmark(agent, n_steps)

    try:
        fp = open("result_table.csv", "r")
        fp.close()
    except Exception:
        print("Initiating result_table.csv")
        fp = open("result_table.csv", "w")
        fp.write("agent,env,seed,nsteps,reward\n")
        fp.close()

    fp = open("result_table.csv", "a")
    for env in result.keys():
        reward = result[env]['total_reward']
        line = ",".join([name, env, str(seed), str(n_steps), str(reward)])
        line += "\n"
        fp.write(line)
    fp.close()

    values = result.values()
    rewards = [x['total_reward'] for x in values]
    avg_reward = sum(rewards)/(len(rewards)*n_steps)
    print("Result: "+name+" got avg reward: " + str(avg_reward))

def seeded_A2C(*args, **kwargs):
    return agent_A2C(*args, seed=seed, **kwargs)
def seeded_PPO(*args, **kwargs):
    return agent_PPO(*args, seed=seed, **kwargs)
def seeded_DQN(*args, **kwargs):
    return agent_DQN(*args, seed=seed, **kwargs)

agents = [
    ['random_agent', random_agent, None],
    ['constant_agent', constant_agent, None],
    ['naive_learner', naive_learner, None],
    ['agent_A2C', seeded_A2C, clear_cache_A2C],
    ['agent_DQN', seeded_DQN, clear_cache_DQN],
    ['agent_PPO', seeded_PPO, clear_cache_PPO],
]

for (name, agent, cache_clear_fnc) in agents:
    measure_agent(name, agent)
    name = "reality_check("+name+")"
    agent = reality_check(agent)
    measure_agent(name, agent)
    if cache_clear_fnc is not None:
        cache_clear_fnc()