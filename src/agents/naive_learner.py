import random

from util import memoize

@memoize
def naive_learner(prompt, num_legal_actions, num_possible_obs):
    reward_lists = {i:() for i in range(num_legal_actions)}

    if random.random()<.15:
        return int(random.random()*num_legal_actions)

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