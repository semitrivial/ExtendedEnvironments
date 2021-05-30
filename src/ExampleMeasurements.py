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
# Code for measuring these agents is in example.py. Commandline arguments
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
# compiled the table in the paper using the script "stats.q".
#
# If you wish, you can run this file you're reading now and have it perform
# the above measurements for you. In order for the script to work, you will
# need to have first followed the installation instructions to activate the
# library's prerequisite, Stable Baselines3. See README.md for details about
# how to do this.
#
import os

print("Deleting result_table.csv (if it exists)...")
os.system("rm result_table.csv")

print("Measuring agents for 500 steps, 10 different seeds...")

for seed in range(10):
    os.system("python example.py steps 500 seed "+str(seed))

print("Measuring agents for 1000 steps, 10 different seeds...")

for seed in range(10):
    os.system("python example.py steps 1000 seed "+str(seed))

print("Done.")
print("""
    Results should now be written in result_table.csv,
    and the table in the paper can be compiled using
    stats.q
""")