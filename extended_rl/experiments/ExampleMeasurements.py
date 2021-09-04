# This file contains instructions for how to replicate the
# numerical results in Section 5 of the paper:
# "Extending Environments To Measure Self-Reflection In Reinforcement Learning"
#
# The six agents measured in the paper are defined in the agents directory
# (misc_agents.py, naive_learner.py, SBL3_agents.py). The environments on
# which the agents are run are defined in the environments directory. Agents
# are also run on the corresponding opposite environments (the opposite of
# an environment is the environment which results by multiplying all rewards
# by -1).
#
# Code for measuring these agents is in experiment.py. Commandline arguments
# can be used to specify a random seed and a number of steps. Each agent
# is run in each environment for the specified number of steps, and the
# results are appended to result_table.csv. This CSV file comes already
# populated with results; one should delete result_table.csv if one wants
# to begin measurement from scratch. For each agent A, for each environment
# E, a row is appended to result_table.csv specifying:
# * The agent
# * The environment
# * The number of steps
# * The random seed
# * The average reward the agent earned in the environment (per turn)
#
# For the purpose of the paper, the agents were run against the environments
# for 500 steps, repeated 10 times with random seeds 0,1,...,9. The agents
# were also run against the environments for 1000 steps, repeated 10 times
# with random seeds 0,1,...,9. From the resulting result_table.csv, we
# compiled the table in the paper using the script "stats.py".
#
# If you wish, you can run this file you're reading now and have it perform
# the above measurements for you. In order for the script to work, you will
# need to have first followed the installation instructions to activate the
# library's prerequisite, Stable Baselines3. See README.md for details about
# how to do this. Assuming the prerequisite is installed, to run this script
# you should go to the src directory and type:
#   python -m experiments.ExampleMeasurements
#
import os

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
from environments.MinusRewards import minus_rewards

agents = {
    'RandomAgent': RandomAgent,
    'ConstantAgent': ConstantAgent,
    'NaiveLearner': NaiveLearner,
    'Q_learner': Q_learner,
    'DQN_learner': DQN_learner,
    'PPO_learner': PPO_learner,
    'A2C_learner': A2C_learner,
}
rcs = [reality_check(x) for x in agents.values()]
agents.update({rc.__name__: rc for rc in rcs})

envs = {x.__name__: x for x in environments}
composed = [compose_envs(v,e) for v in vanilla_envs for e in environments]
envs.update({c.__name__: c for c in composed})
minus = [minus_rewards(e) for e in envs.values()]
envs.update({m.__name__: m for m in minus})

seeds = [0, 1, 2, 3, 4]

total_tasks = 0
steps = 10
for seed in seeds:
    for agent in agents.keys():
        for env in envs.keys():
            total_tasks += 1

def run_task(seed, agent, env, n):
    print(f"Task {n} out of {total_tasks}:")
    name_a = agent.replace("(", "_").replace(")", "_")
    name_e = env.replace("(", "_").replace(")", "_")
    filename = f"../../extended_rl_results/{seed}_{name_a}_{name_e}.csv"
    os.system(
        f"python -m experiments.experiment steps {steps} seed {seed} agent {agent} env {env} logfile {filename}"
    )
    print("Task {n} completed.")

#print("Deleting result_table.csv (if it exists)...")
#os.system("rm experiments/result_table.csv")

starting_task = 0
steps = 10
n = 0
for seed in seeds:
    for agent in agents.keys():
        for env in envs.keys():
            if n >= starting_task:
                run_task(seed, agent, env, n)
            n += 1


print("Done.")
print("""
    Results should now be written in result_table.csv,
    and the table in the paper can be compiled using
    stats.py
""")