from util import fast_run_env
from agents.Q import Q_learner
from environments.IgnoreRewards import IgnoreRewards

A = Q_learner(epsilon=0.9, alpha=0.1, gamma=0.9)
env = IgnoreRewards
n_steps = 1000000

results = fast_run_env(env, A, n_steps)
print("Avg Reward: "+str(results['total_reward']/n_steps))