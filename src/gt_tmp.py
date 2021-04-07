from agents.SBL3_agents import agent_A2C, agent_DQN, agent_PPO
from IgnoreRewards import ignore_rewards
from GuardedTreasures import GuardedTreasures
from util import run_environment
from BackwardConsciousness import BackwardConsciousness
from CryingBaby import CryingBaby
from IgnoreRewards import IgnoreRewards

#e = GuardedTreasures
#e = BackwardConsciousness
e = CryingBaby

N = 400
import cProfile
cProfile.run("run_environment(e, agent_DQN, N)")
#result = run_environment(e, agent_DQN, N)
#print('Total reward ' + str(result['total_reward']/N))
