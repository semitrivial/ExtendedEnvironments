# Run this script by typing the following in the src directory:
# python -m experiments.TemptingButtonExperiment
import random

import numpy as np
import torch

from agents.custom_DQN import custom_DQN_agent
from TemptingButton import TemptingButton
from util import run_environment

n_steps = 4000
seed = 0

np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)

print("Running recurrent DQN agent on TemptingButton...")
result = run_environment(TemptingButton, custom_DQN_agent, n_steps)
print("...Avg reward: "+str(result['total_reward']/n_steps))

# Various variations with different hyperparameters

def run(variation, descr):
    print("Running agent ("+descr+") on TemptingButton")
    result = run_environment(TemptingButton, variation, n_steps)
    print("...Avg reward: "+str(result['total_reward']/n_steps))

def variation1(*args):
    return custom_DQN_agent(*args, learning_rate=.05)
run(variation1, "with learning_rate=.05")

def variation2(*args):
    return custom_DQN_agent(*args, learning_rate=.1)
run(variation2, "with learning_rate=.1")

def variation3(*args):
    return custom_DQN_agent(*args, learning_rate=1e-2)
run(variation3, "with learning_rate=1e-2")

def variation4(*args):
    return custom_DQN_agent(*args, learning_rate=1e-3)
run(variation4, "with learning_rate=1e-3")

def variation5(*args):
    return custom_DQN_agent(*args, lookback=1)
run(variation5, "with lookback=1")

def variation6(*args):
    return custom_DQN_agent(*args, lookback=3)
run(variation6, "with lookback=3")

def variation7(*args):
    return custom_DQN_agent(*args, lookback=5)
run(variation7, "with lookback=5")