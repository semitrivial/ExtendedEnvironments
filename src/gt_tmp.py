from agents.SBL3_agents import agent_A2C, agent_DQN, agent_PPO
from IgnoreRewards import ignore_rewards
from GuardedTreasures import GuardedTreasures
from util import run_environment
from BackwardConsciousness import BackwardConsciousness
from HistoryObservable import HistoryObservable

e = HistoryObservable(GuardedTreasures)

result = run_environment(e, agent_DQN, 1000)
print(result['total_reward'])
