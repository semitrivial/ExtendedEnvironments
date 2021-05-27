# Run this script by typing the following in the src directory:
# python -m experiments.TemptingButtonExperiment

from agents.custom_DQN import custom_DQN_agent
from TemptingButton import TemptingButton
from util import run_environment

n_steps = 1000

print("Running recurrent DQN agent on TemptingButton")
result = run_environment(TemptingButton, custom_DQN_agent, n_steps)
print("Avg reward: "+str(result['total_reward']/n_steps))