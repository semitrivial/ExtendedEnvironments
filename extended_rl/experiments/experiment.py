# Script used to generate Table 1 in "Extending Environments To
# Measure Self-Reflection In Reinforcement Learning".
# For instructions, see ExampleMeasurements.py.
import sys
import random
from collections import deque

#import numpy as np
#import torch

from agents.Q import Q_learner
from agents.recurrent_Q import recurrent_Q
from agents.SBL3_DQN import DQN_learner
from agents.SBL3_A2C import A2C_learner
from agents.SBL3_PPO import PPO_learner
from agents.misc_agents import RandomAgent, ConstantAgent
from agents.naive_learner import NaiveLearner1, NaiveLearner2, NaiveLearner3, NaiveLearner4
from agents.reality_check import reality_check
from selfreflection_benchmark import selfrefl_benchmark
from util import memoize
from prerandom import populate_randoms

seed, n_steps = 0, 100

# Parse command-line arguments
args = deque(sys.argv[1:])
while args:
    arg = args.popleft()
    if arg == 'seed':
        seed = int(args.popleft())
    elif arg == 'steps':
        n_steps = int(args.popleft())
    else:
        raise ValueError("Unrecognized commandline argument")

populate_randoms(seed)

print(f"Testing agents with seed={seed}, n_steps={n_steps}")

#np.random.seed(seed)
#torch.manual_seed(seed)

def measure_agent(name, agent):
    print(f"Testing {name}...")
    result = selfrefl_benchmark(agent, n_steps)

    # If result_table.csv does not already exist, then create it and
    # write headers to it.
    try:
        fp = open("experiments/result_table.csv", "r")
        fp.close()
    except Exception:
        print("Initiating result_table.csv")
        fp = open("experiments/result_table.csv", "w")
        fp.write("agent,env,seed,nsteps,reward\n")
        fp.close()

    # Append rows to result_table.csv, one row per environment, indicating
    # how the given agent performed in those environments.
    fp = open("experiments/result_table.csv", "a")
    for env in result.keys():
        reward = result[env]['total_reward']
        line = ",".join([name, env, str(seed), str(n_steps), str(reward)])
        line += "\n"
        fp.write(line)
    fp.close()

    values = result.values()
    rewards = [x['total_reward'] for x in values]
    avg_reward = sum(rewards)/(len(rewards)*n_steps)
    print(f"Result: {name} got avg reward: {avg_reward}")

agents = [
    # ['RandomAgent', RandomAgent],
    # ['ConstantAgent', ConstantAgent],
    # ['NaiveLearner1', NaiveLearner1],
    # ['NaiveLearner2', NaiveLearner2],
    # ['NaiveLearner3', NaiveLearner3],
    # ['NaiveLearner4', NaiveLearner4],
    # ['Q_learner', Q_learner],
    # ['Recurrent Q', recurrent_Q],
    ['DQN', DQN_learner],
    # ['PPO', PPO_learner],
    # ['A2C', A2C_learner],
]

# Measure all the above-listed agents and their reality-checks
for (name, agent) in agents:
    measure_agent(name, agent)
    name = f"reality_check({name})"
    agent = reality_check(agent)
    measure_agent(name, agent)