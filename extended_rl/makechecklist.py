from environments.EnvironmentLists import environments
from agents.Q import Q_learner
from agents.SBL3_DQN import DQN_learner
from agents.SBL3_A2C import A2C_learner
from agents.SBL3_PPO import PPO_learner
from agents.misc_agents import RandomAgent, ConstantAgent
from agents.naive_learner import NaiveLearner

agents = [
    RandomAgent,
    ConstantAgent,
    NaiveLearner,
    Q_learner,
    DQN_learner,
    PPO_learner,
    A2C_learner
]

finished = {
    ConstantAgent: [0,1,2,3,4],
    RandomAgent: [0,1,2,3,4],
    NaiveLearner: [0,1,2,3,4],
    Q_learner: [0,1,2,3,4],
}

# DQN plain: Done
# DQN_RC plain:
# DQN composed w/vanilla:
# DQN_RC composed w/vanilla:


seeds = [0, 1, 2, 3, 4]

fp = open("experiments/checklist.txt", "w")

for env in environments:
    for agent in agents:
        for seed in seeds:
            if agent in finished and seed in finished[agent]:
                fp.write(f"{env.__name__}, {agent.__name__}, {seed}: Done\n")
            else:
                fp.write(f"{env.__name__}, {agent.__name__}, {seed}\n")

fp.close()