from test.ad_hoc_tests import run_ad_hoc_tests
from test.test_agents import agents
from EnvironmentLists import environments
from util import run_environment

run_ad_hoc_tests()

def test_environment(env, env_name):
    for agent_name in agents.keys():
        agent = agents[agent_name]
        results = run_environment(env, agent, 100)
        print("Reward for "+agent_name+" in "+env_name+": "+str(results['total_reward']))

for name, env in environments.items():
    test_environment(env, name)
