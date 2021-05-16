import sys
import random
from collections import deque

import numpy as np
import torch

from agents.SBL3_agents import (
    agent_A2C, agent_DQN, agent_PPO,
    clear_cache_A2C, clear_cache_DQN, clear_cache_PPO
)
from RealityCheck import reality_check
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

@memoize
def random_agent(prompt, num_legal_actions, num_possible_obs):
    return int(random.random() * num_legal_actions)

def constant_agent(prompt, num_legal_actions, num_possible_obs):
    return 0

@memoize
def naive_learner(prompt, num_legal_actions, num_possible_obs):
    reward_lists = {i:() for i in range(num_legal_actions)}

    if random.random()<.15:
        return int(random.random()*num_legal_actions)

    for i in range(len(prompt)):
        is_reward = (i%3)==0
        if is_reward and i>0:
            reward = prompt[i]
            prev_action = prompt[i-1]
            reward_lists[prev_action] = reward_lists[prev_action] + (reward,)

    avg_rewards = {x:float(sum(y))/(1+len(y)) for x,y in reward_lists.items()}
    best_reward = -99999
    for x,y in avg_rewards.items():
        if y > best_reward:
            best_reward = y
            best_action = x

    return best_action

def measure_agent(name, agent):
    print("Testing "+name+"...")
    n_steps = 50
    result = awareness_benchmark(agent, n_steps)

    try:
        fp = open("result_table.csv", "r")
        fp.close()
    except Exception:
        print("Initiating results_table.csv")
        fp = open("result_table.csv", "w")
        fp.write("agent,env,nsteps,reward\n")
        fp.close()

    fp = open("result_table.csv", "a")
    for env in result.keys():
        reward = result[env]['total_reward']
        line = ",".join([name, env, str(n_steps), str(reward)])
        line += "\n"
        fp.write(line)
    fp.close()

    values = result.values()
    rewards = [x['total_reward'] for x in values]
    avg_reward = sum(rewards)/(len(rewards)*n_steps)
    print("Result: "+name+" got avg reward: " + str(avg_reward))

def seeded_A2C(prompt, num_legal_actions, num_possible_obs, **kwargs):
    return agent_A2C(prompt, num_legal_actions, num_possible_obs, seed=seed)
def seeded_PPO(prompt, num_legal_actions, num_possible_obs, **kwargs):
    return agent_PPO(prompt, num_legal_actions, num_possible_obs, seed=seed)
def seeded_DQN(prompt, num_legal_actions, num_possible_obs, **kwargs):
    return agent_DQN(prompt, num_legal_actions, num_possible_obs, seed=seed)

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