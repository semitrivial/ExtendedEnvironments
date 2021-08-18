# The purpose of this file is to test the library to make sure
# it works. End-users who are not working on contributing code
# to the library do not need to worry about this.

#from test.ad_hoc_tests import run_ad_hoc_tests
from test.test_util import test_util
#from test.test_agents import agents
from agents.Q import Q_learner
from environments.EnvironmentLists import environments
from util import run_environment

test_util()
#run_ad_hoc_tests()

def test_environment(env, env_name):
    agent = agents[agent_name]
    results = run_environment(env, agent, 100)

print("Running all environments...")

for name, env in environments.items():
    run_environment(env, Q_learner, 1000)

print("Done.")