from random import random
from pprint import pprint

# To show the library in action, we need an agent to test. The following
# is a simple agent for that purpose.

def simple_agent(prompt, num_legal_actions, num_possible_obs):
    """
    A simple agent which repeats the past action which most recently yielded
    a positive immediate reward when taken in response to the same
    observation. If no such past action exists, the agent acts randomly.
    """
    latest_observ = prompt[-1]
    disqualified = []

    # Since the next sequence-element to be added to prompt---which
    # will have index len(prompt)---will be the action the agent is
    # currently computing, it follows that the past actions (in
    # reverse-order) have indices len(prompt)-3, len(prompt)-6, etc.
    for action_index in range(len(prompt)-3, 0, -3):
        past_action = prompt[action_index]
        past_observ = prompt[action_index-1]
        past_reward = prompt[action_index+1]

        if (past_observ != latest_observ) or (past_action in disqualified):
            continue
        if past_reward <= 0:
            disqualified.append(past_action)
            continue

        # We found an action which, based solely on the most recent
        # available evidence, yields positive reward. Take that action.
        return past_action

    # No actions identified which yield positive reward based solely
    # on most recent available evidence. As fallback, act randomly.
    return int(random() * num_legal_actions)

from extended_rl import selfrefl_benchmark
from extended_rl import selfrefl_measure

print("Results of running agent for 100 steps in various environments:")
pprint(selfrefl_benchmark(simple_agent, 100))

print("---------------------------------------------------------------")
avg_reward = selfrefl_measure(simple_agent, 100)
print("Average reward per turn: " + str(avg_reward))
