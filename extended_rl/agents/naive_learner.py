import random

from extended_rl.util import memoize

@memoize
def naive_learner(prompt, num_legal_actions, num_possible_obs):
    """
    Agent which acts randomly 15% of the time, and otherwise chooses
    the action which has historically resulted in the largest average
    immediate reward. Agent ignores observations. If an action was
    never taken before in the given prompt, then its average immediate
    reward is considered to be 0. If multiple actions are tied for
    having the largest average immediate reward, the tie is broken
    based on python dictionary order (which might be non-determinstic,
    depending on the python version, but the agent is memoized, so the
    same prompt will always output the same response, and the agent is
    therefore ultimately deterministic).
    """
    if random.random()<.15:
        return int(random.random()*num_legal_actions)

    # Dictionary for associating with each action the tuple of
    # immediate rewards which followed that action
    reward_lists = {i:() for i in range(num_legal_actions)}

    # Populate the above dictionary
    for i in range(len(prompt)):
        is_reward = (i%3)==0
        if is_reward and i>0:
            reward = prompt[i]
            prev_action = prompt[i-1]
            reward_lists[prev_action] = reward_lists[prev_action] + (reward,)

    avg_rewards = {x:float(sum(y))/(1+len(y)) for x,y in reward_lists.items()}
    best_reward = -99999
    for x,y in avg_rewards.items():
        if y > best_reward:
            best_reward = y
            best_action = x

    return best_action