from agents.custom_DQN import custom_DQN_agent
from GuardedTreasures import GuardedTreasures # NOTE: NOT the file from your branch
from util import run_environment

if __name__ == '__main__':
	N = 10000
	result = run_environment(GuardedTreasures, custom_DQN_agent, N)
	print('Total reward/N: ' + str(result['total_reward']/N))

def main(n=20000):
	N = n
	print(N)
	result = run_environment(GuardedTreasures, custom_DQN_agent, N)
	print('Total reward/N: ' + str(result['total_reward']/N))

	return result

