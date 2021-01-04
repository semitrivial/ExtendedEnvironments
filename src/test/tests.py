from test_agents import agents
from IncentivizeZero import incentivize_zero
from GuardedTreasures import guarded_treasures
from util import run_environment

def test_incentivize_zero():
    for name in agents.keys():
        agent = agents[name]
        results = run_environment(incentivize_zero, agent, 100)
        print("Results for "+name+" in incentivize_zero: " + str(results['total_reward']))

def test_guarded_treasures():
    for name in agents.keys():
        agent = agents[name]
        results = run_environment(guarded_treasures, agent, 100)
        print("Results for "+name+" in guarded_treasures: " + str(results['total_reward']))