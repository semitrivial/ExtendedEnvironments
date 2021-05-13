from random import random
from collections import OrderedDict

from agents.SBL3_agents import agent_A2C, agent_DQN, agent_PPO
from agents.custom_DQN import custom_DQN_agent
from RealityCheck import reality_check
from AwarenessBenchmark import awareness_benchmark
from util import cache


@cache
def random_agent(prompt, num_legal_actions, num_possible_obs):
    return int(random() * num_legal_actions)

@cache
def incrementer(prompt, num_legal_actions, num_possible_obs):
    return ((len(prompt)+1)/3)%num_legal_actions

@cache
def always_0(prompt, num_legal_actions, num_possible_actions):
    return 0

@cache
def always_1(prompt, num_legal_actions, num_possible_actions):
    return 1

@cache
def naive_learner(prompt, num_legal_actions, num_possible_actions):
    reward_lists = {i:() for i in range(num_legal_actions)}

    if random()<.15:
        return int(random()*num_legal_actions)

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
    ['random_agent', random_agent],
    ['incrementer', incrementer],
    ['always_0', always_0],
    ['always_1', always_1],
    ['naive_learner', naive_learner],
    ['agent_A2C', agent_A2C],
    ['agent_DQN', agent_DQN],
    ['agent_PPO', agent_PPO],
    ['custom_DQN_agent', custom_DQN_agent],
])

def measure_agent(name, agent):
    print("Testing "+name+"...")
    n_steps = 100
    result = awareness_benchmark(agent, n_steps)

    try:
        fp = open("result_table.csv", "r")
        fp.close()
    except Exception:
        print("Initiating results_table.csv")
        fp = open("result_table.csv", "w")
        fp.write("agent,env,nsteps,reward\n")
        fp.close()

    fp = open("result_table.csv", "a")
    for env in result.keys():
        reward = result[env]['total_reward']
        line = ",".join([name, env, str(n_steps), str(reward)])
        line += "\n"
        fp.write(line)
    fp.close()

    values = result.values()
    rewards = [x['total_reward'] for x in values]
    avg_reward = sum(rewards)/(len(rewards)*n_steps)
    print("Result: "+name+" got avg reward: " + str(avg_reward))

for name, agent in agents.items():
    measure_agent(name, agent)
    name = "reality_check("+name+")"
    agent = reality_check(agent)
    measure_agent(name, agent)
