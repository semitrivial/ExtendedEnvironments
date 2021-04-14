from agents.custom_DQN import custom_DQN_agent
from GuardedTreasures import GuardedTreasures # NOTE: NOT the file from your branch
from util import run_environment

N = 10000
result = run_environment(GuardedTreasures, custom_DQN_agent, N)
print('Total reward/N: ' + str(result['total_reward']/N))
