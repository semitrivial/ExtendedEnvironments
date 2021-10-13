import random
from pprint import pprint

random.seed(0)

# To show the library in action, we need an agent to test. The following
# is a simple agent for that purpose. This takes the first available action
# which has not previously yielded a punishment for the observation in
# question (or action 0 if there is no such action).
class SimpleAgent:
    def __init__(self, **kwargs):
        self.was_action_punished = {
            (o, a): False
            for o in range(self.n_obs)
            for a in range(self.n_actions)
        }

    def act(self, obs):
        for action in range(self.n_actions):
            if not(self.was_action_punished[obs, action]):
                return action
        return 0

    def train(self, o_prev, a, r, o_next):
        if r < 0:
            self.was_action_punished[o_prev, a] = True


from extended_rl import selfrefl_benchmark
from extended_rl import selfrefl_measure

print("Results of running agent for 100 steps in various environments:")
pprint(selfrefl_benchmark(SimpleAgent, 100))

print("---------------------------------------------------------------")
avg_reward = selfrefl_measure(SimpleAgent, 100)
print("Average reward per turn: " + str(avg_reward))
