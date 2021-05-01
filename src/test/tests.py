from test.test_agents import agents
from util import run_environment

from IncentivizeZero import IncentivizeZero
from GuardedTreasures import GuardedTreasures
from IgnoreRewards import IgnoreRewards
from DejaVu import DejaVu
from CryingBaby import CryingBaby
from FalseMemories import FalseMemories
from BackwardConsciousness import BackwardConsciousness
from RuntimeInspector import PunishSlowAgent, PunishFastAgent
from ThirdActionForbidden import ThirdActionForbidden

def test_environment(env, env_name):
    for agent_name in agents.keys():
        agent = agents[agent_name]
        results = run_environment(env, agent, 100)
        print("Reward for "+agent_name+" in "+env_name+": "+str(results['total_reward']))

def test_incentivize_zero():
    test_environment(IncentivizeZero, "incentivize_zero")

def test_guarded_treasures():
    test_environment(GuardedTreasures, "guarded_treasures")

def test_deja_vu():
    test_environment(DejaVu, "deja_vu")

def test_crying_baby():
    test_environment(CryingBaby, "crying_baby")

def test_ignore_rewards():
    test_environment(IgnoreRewards, "ignore_rewards")

def test_false_memories():
    test_environment(FalseMemories, "false_memories")

def test_backward_consciousness():
    test_environment(BackwardConsciousness, "backward_consciousness")

def test_punish_slow_agent():
    test_environment(PunishSlowAgent, "punish_slow_agent")

def test_punish_fast_agent():
    test_environment(PunishFastAgent, "punish_fast_agent")

def test_third_action_forbidden():
    test_environment(ThirdActionForbidden, "third_action_forbidden")