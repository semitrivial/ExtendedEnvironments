# The purpose of this file is to test the library to make sure
# it works. End-users who are not working on contributing code
# to the library do not need to worry about this.

from test.ad_hoc_tests import run_ad_hoc_tests
from test.test_util import test_util
from agents.Q import Q_learner
from agents.misc_agents import RandomAgent, ConstantAgent
from environments.EnvironmentLists import environments
from test.monkeypatches import run_environment

test_util()
run_ad_hoc_tests()

agents = [Q_learner, RandomAgent, ConstantAgent]

print("Running all environments...")

for env in environments:
    for agent in agents:
        run_environment(env, agent, 100)

print("Done.")