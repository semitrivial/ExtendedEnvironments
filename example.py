from random import random
from pprint import pprint

# To show the library in action, we need an agent to test. The following
# is a simple agent for that purpose. This agent acts randomly unless
# either the last observation was 0 or the last reward was 0.
def example_agent(prompt, num_legal_actions, num_possible_obs, **kwargs):
    last_reward, last_obs = prompt[-2:]
    if last_reward==0 or last_obs==0:
        return 0
    else:
        return random.randrange(num_legal_actions)

from extended_rl import selfrefl_benchmark
from extended_rl import selfrefl_measure

print("Results of running agent for 100 steps in various environments:")
pprint(selfrefl_benchmark(example_agent, 100))

print("---------------------------------------------------------------")
avg_reward = selfrefl_measure(example_agent, 100)
print("Average reward per turn: " + str(avg_reward))
