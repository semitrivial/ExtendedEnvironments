from util import run_environment
from agents.Q import Q_learner
from agents.recurrent_Q import recurrent_Q
from agents.reality_check import reality_check
from agents.SBL3_PPO import PPO_learner
from agents.SBL3_A2C import A2C_learner
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

class TestEnv:
    def __init__(self, A):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        return
    def start(self):
        return 0
    def step(self, action):
        reward = 1 if action==0 else -1
        return (reward, 0)

A = A2C_learner
env = TestEnv
n_steps = 1000

print("Without reality_check:")
results = run_environment(env, A, n_steps)
print("Avg Reward: "+str(results['total_reward']/n_steps))

# print("With reality check:")
# results = run_environment(env, reality_check(A), n_steps)
# print("Avg Reward: "+str(results['total_reward']/n_steps))

# print("With minus_rewards:")
# results = run_environment(minus_rewards(env), A, n_steps)
# print("Avg Reward: "+str(results['total_reward']/n_steps))

# print("With minus_rewards and reality check:")
# results = run_environment(minus_rewards(env), reality_check(A), n_steps)
# print("Avg Reward: "+str(results['total_reward']/n_steps))