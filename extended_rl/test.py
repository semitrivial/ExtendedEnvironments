from test.ad_hoc_tests import run_ad_hoc_tests
from test.test_agents import agents
from environments.EnvironmentLists import environments
from util import run_environment

run_ad_hoc_tests()

def test_environment(env, env_name):
    for agent_name in agents.keys():
        agent = agents[agent_name]
        results = run_environment(env, agent, 100)

print("Running all environments...")

for name, env in environments.items():
    test_environment(env, name)

print("Done.")