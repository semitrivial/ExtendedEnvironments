from test_agents import agents
from util import run_environment

from IncentivizeZero import incentivize_zero
from GuardedTreasures import guarded_treasures
from IgnoreRewards import ignore_rewards
from SelfInsert import self_insert
from DejaVu import deja_vu

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

def test_ignore_rewards():
    test_environment(ignore_rewards(incentivize_zero), "ignore_rewards(incentivize_zero)")
    test_environment(ignore_rewards(guarded_treasures), "ignore_rewards(guarded_treasures)")
    test_environment(ignore_rewards(deja_vu), "ignore_rewards(deja_vu)")

def test_self_insert():
    test_environment(self_insert(incentivize_zero), "self_insert(incentivize_zero)")
    test_environment(self_insert(guarded_treasures), "self_insert(guarded_treasures)")
    test_environment(self_insert(deja_vu), "self_insert(deja_vu)")