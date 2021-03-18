from test_agents import agents
from util import run_environment

from IncentivizeZero import incentivize_zero
from GuardedTreasures import guarded_treasures
from IgnoreRewards import ignore_rewards
from DejaVu import deja_vu
from CryingBaby import crying_baby
from FalseMemories import false_memories
from BackwardConsciousness import backward_consciousness
from RuntimeInspector import punish_slow_agent, punish_fast_agent

def test_environment(env, env_name):
    for agent_name in agents.keys():
        agent = agents[agent_name]
        results = run_environment(env, agent, 100)
        print("Reward for "+agent_name+" in "+env_name+": "+str(results['total_reward']))

def test_incentivize_zero():
    test_environment(incentivize_zero, "incentivize_zero")

def test_guarded_treasures():
    test_environment(guarded_treasures, "guarded_treasures")

def test_deja_vu():
    test_environment(deja_vu, "deja_vu")

def test_crying_baby():
    test_environment(crying_baby, "crying_baby")

def test_ignore_rewards():
    test_environment(ignore_rewards, "ignore_rewards")

def test_false_memories():
    test_environment(false_memories, "false_memories")

def test_backward_consciousness():
    test_environment(backward_consciousness, "backward_consciousness")

def test_punish_slow_agent():
    test_environment(punish_slow_agent, "punish_slow_agent")

def test_punish_fast_agent():
    test_environment(punish_fast_agent, "punish_fast_agent")