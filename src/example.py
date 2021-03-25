from random import random
from collections import OrderedDict

from AwarenessBenchmark import awareness_benchmark

def narrow_random_agent(prompt):
    return int(random()*3)

def wide_random_agent(prompt):
    return int(random()*10)

def narrow_incrementer(prompt):
    return ((len(prompt)+1)/3)%3

def wide_incrementer(prompt):
    return ((len(prompt)+1)/3)%10

def always_0(prompt):
    return 0

def always_1(prompt):
    return 1

def naive_learner(prompt):
    reward_lists = {i:() for i in range(10)}

    if random()<.15:
        return int(random()*10)

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

agents = OrderedDict([
    ['narrow_random_agent', narrow_random_agent],
    ['wide_random_agent', wide_random_agent],
    ['narrow_incrementer', narrow_incrementer],
    ['wide_incrementer', wide_incrementer],
    ['always_0', always_0],
    ['always_1', always_1],
    ['naive_learner', naive_learner]
])

from test.SB3_agents import get_SB3_agents
try:
    SB3_agents = get_SB3_agents()
    agents.update(SB3_agents)
except ImportError:
    print("-------------")
    print("Skipping SB3 agents because could not import dependencies")
    print("-------------")

for name, agent in agents.items():
    print("Testing "+name+"...")
    result = awareness_benchmark(agent, 100)
    rewards = result.values()
    avg_reward = float(sum(rewards))/len(rewards)
    print("Result: "+name+" got avg reward "+str(avg_reward))