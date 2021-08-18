from util import fast_run_env
from agents.Q import Q_learner
from agents.reality_check import reality_check
from environments.MinusRewards import minus_rewards
from environments.IgnoreRewards import IgnoreRewards
from environments.IgnoreRewards2 import IgnoreRewards2
from environments.IgnoreRewards3 import IgnoreRewards3
from environments.IgnoreActions import IgnoreActions
from environments.IgnoreObservations import IgnoreObservations
from environments.IncentivizeLearningRate import IncentivizeLearningRate
from environments.IncentivizeZero import IncentivizeZero
from environments.LimitedMemory import LimitedMemory
from environments.Repeater import Repeater
from environments.RuntimeInspector import PunishSlowAgent, PunishFastAgent
from environments.ShiftedRewards import ShiftedRewards

A = Q_learner
env = ShiftedRewards
n_steps = 10000

print("Without reality_check:")
results = fast_run_env(env, A, n_steps)
print("Avg Reward: "+str(results['total_reward']/n_steps))

print("With reality check:")
results = fast_run_env(env, reality_check(A), n_steps)
print("Avg Reward: "+str(results['total_reward']/n_steps))

print("With minus_rewards:")
results = fast_run_env(minus_rewards(env), A, n_steps)
print("Avg Reward: "+str(results['total_reward']/n_steps))

print("With minus_rewards and reality check:")
results = fast_run_env(minus_rewards(env), reality_check(A), n_steps)
print("Avg Reward: "+str(results['total_reward']/n_steps))