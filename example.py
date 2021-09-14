import random
from pprint import pprint

random.seed(0)

# To show the library in action, we need an agent to test. The following
# is a simple agent for that purpose. This agent acts randomly unless
# either the last observation was 0 or the last reward was 0.
class ExampleAgent:
    def __init__(self):
        self.prev_obs_was_0 = False
        self.prev_reward_was_0 = False

    def act(self, obs):
        if self.prev_obs_was_0 or self.prev_reward_was_0:
            return 0
        else:
            return random.randrange(self.num_legal_actions)

    def train(self, o_prev, a, r, o_next):
        self.prev_obs_was_0 = (o_next == 0)
        self.prev_reward_was_0 = (r == 0)


from extended_rl import selfrefl_benchmark
from extended_rl import selfrefl_measure

print("Results of running agent for 100 steps in various environments:")
pprint(selfrefl_benchmark(example_agent, 100))

print("---------------------------------------------------------------")
avg_reward = selfrefl_measure(example_agent, 100)
print("Average reward per turn: " + str(avg_reward))
