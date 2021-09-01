from util import run_environment
from agents.Q import Q_learner
from agents.recurrent_Q import recurrent_Q
from agents.reality_check import reality_check
from agents.SBL3_PPO import PPO_learner
from agents.SBL3_A2C import A2C_learner
from agents.SBL3_DQN import DQN_learner
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
from environments.TemptingButtonVariation import TemptingButtonVariation
from environments.TemptingButton import TemptingButton
from environments.composition import compose_envs
from environments.Vanilla import SimplePath

A = Q_learner
env = compose_envs(SimplePath, IgnoreObservations)
n_steps = 100000

print("Without reality_check:")
results = run_environment(env, A, n_steps)
print("Avg Reward: "+str(results['total_reward']/n_steps))

print("With reality check:")
results = run_environment(env, reality_check(A), n_steps)
print("Avg Reward: "+str(results['total_reward']/n_steps))

print("With minus_rewards:")
results = run_environment(minus_rewards(env), A, n_steps)
print("Avg Reward: "+str(results['total_reward']/n_steps))

print("With minus_rewards and reality check:")
results = run_environment(minus_rewards(env), reality_check(A), n_steps)
print("Avg Reward: "+str(results['total_reward']/n_steps))