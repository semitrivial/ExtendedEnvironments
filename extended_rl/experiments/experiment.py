# Script used to generate Table 1 in "Extending Environments To
# Measure Self-Reflection In Reinforcement Learning".
# For instructions, see ExampleMeasurements.py.
import sys
import random
from collections import deque

import numpy as np
import torch

from agents.Q import Q_learner
from agents.SBL3_DQN import DQN_learner
from agents.SBL3_A2C import A2C_learner
from agents.SBL3_PPO import PPO_learner
from agents.misc_agents import RandomAgent, ConstantAgent
from agents.naive_learner import NaiveLearner
from agents.reality_check import reality_check
from environments.EnvironmentLists import environments
from environments.Vanilla import vanilla_envs
from environments.composition import compose_envs
from extended_rl.environments.MinusRewards import minus_rewards
from util import run_environment, args_to_agent
from prerandom import populate_randoms

seed, n_steps = 0, 100
logfile = None
agent_name = None
env_name = None

# Parse command-line arguments
args = deque(sys.argv[1:])
while args:
    arg = args.popleft()
    if arg == 'seed':
        seed = int(args.popleft())
    elif arg == 'steps':
        n_steps = int(args.popleft())
    elif arg == 'logfile':
        logfile = open(args.popleft(), "w")
    elif arg == 'agent':
        agent_name = args.popleft()
    elif arg == 'env':
        env_name = args.popleft()
    else:
        raise ValueError("Unrecognized commandline argument")

if agent_name is None:
    raise ValueError("No agent specified")
if env_name is None:
    raise ValueError("No environment specified")

populate_randoms(seed)
np.random.seed(seed)
torch.manual_seed(seed)

agents = {
    'RandomAgent': RandomAgent,
    'ConstantAgent': ConstantAgent,
    'NaiveLearner': NaiveLearner,
    'Q_learner': Q_learner,
    'DQN_learner': args_to_agent(DQN_learner, seed=seed),
    'PPO_learner': args_to_agent(PPO_learner, seed=seed),
    'A2C_learner': args_to_agent(A2C_learner, seed=seed),
}
rcs = [reality_check(x) for x in agents.values()]
agents.update({rc.__name__: rc for rc in rcs})

agent = agents[agent_name]

envs = {x.__name__: x for x in environments}
composed = [compose_envs(v,e) for v in vanilla_envs for e in environments]
envs.update({c.__name__: c for c in composed})
minus = [minus_rewards(e) for e in envs.values()]
envs.update({m.__name__: m for m in minus})

env = envs[env_name]

print(f"Testing {agent_name} in {env_name}, seed={seed}, n_steps={n_steps}")

result = run_environment(env, agent, n_steps, logfile)

try:
    fp = open("experiments/result_table.csv", "r")
    fp.close()
except Exception:
    print("Initiating result_table.csv")
    fp = open("experiments/result_table.csv", "w")
    fp.write("agent,env,seed,nsteps,reward\n")
    fp.close()

fp = open("experiments/result_table.csv", "a")
reward = result['total_reward']
line = ",".join([agent_name, env_name, str(seed), str(n_steps), str(reward)])
line += "\n"
fp.write(line)
fp.close()

print(f"Result: {agent_name} got total reward: {result['total_reward']}")

if logfile:
    logfile.close()
