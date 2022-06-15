import os

agents = ["DQN", "RC_DQN", "PPO", "RC_PPO"]
xenvs = ["ignorerewards", "incentivize_learning_rate"]
seeds = [1,2,3,4,5,6,7,8,9,10]

total_tasks = len(agents)*len(xenvs)*len(seeds)

def run_task(seed, agent, env, n):
    print(f"Task {n} out of {total_tasks}:")
    os.system(f"python cartpole.py seed {seed} agent '{agent}' xenv '{env}'")
    print(f"Task {n} completed.")

n = 1
for agent in agents:
    for xenv in xenvs:
        for seed in seeds:
            run_task(seed, agent, xenv, n)
            n += 1
