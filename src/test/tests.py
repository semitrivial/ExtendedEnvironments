from test_agents import agents
from IncentivizeZero import incentivize_zero
from util import run_environment

def test_incentivize_zero():
    for name in agents.keys():
        agent = agents[name]
        results = run_environment(incentivize_zero, agent, 100)
        print(results)

