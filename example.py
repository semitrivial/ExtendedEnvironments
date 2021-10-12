import random
from pprint import pprint

random.seed(0)

# To show the library in action, we need an agent to test. The following
# is a simple agent for that purpose. This agent acts randomly unless
# either the last observation was 0 or the last reward was 0.
class ExampleAgent:
    def __init__(self):
        self.prev_obs = 0

    def act(self, obs):
        if self.prev_obs == obs:
            return 0
        else:
            return random.randrange(self.n_actions)

    def train(self, o_prev, a, r, o_next):
        self.prev_obs = o_prev


from extended_rl import selfrefl_benchmark
from extended_rl import selfrefl_measure

print("Results of running agent for 100 steps in various environments:")
pprint(selfrefl_benchmark(ExampleAgent, 100))

print("---------------------------------------------------------------")
avg_reward = selfrefl_measure(ExampleAgent, 100)
print("Average reward per turn: " + str(avg_reward))
