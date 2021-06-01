from math import sqrt

# This file is used for taking the individual numbers in result_table.csv
# and turning them into the numbers in the table from the paper,
# "Extending Environments To Measure Self-Reflection In Reinforcement Learning".
#
# Note, the table was originally computed using the following kdb+/q script, but
# we replaced it with a python script to avoid the proprietary dependency:
"""
/ stats.q
tbl:("SSIIF";enlist ",") 0: `:result_table.csv;
tbl:update scaled_reward:reward%nsteps from tbl;
tbl:select avg scaled_reward by agent, seed, nsteps from tbl;
select reward:avg scaled_reward, stderr:(sdev scaled_reward)%count[i] by agent, nsteps from tbl
"""

print("Analyzing result_table.csv and distilling it into a summary table...")

fp = open("result_table.csv", "r")
fp.readline()  # Ignore header

lines = []
agents = []

while True:
    rawline = fp.readline()

    if rawline == "":
        break

    rawline = rawline[:-1]  # Remove newline
    line = rawline.split(",")
    lines.append(line)
    agent = line[0]

    if not(agent in agents):
        agents.append(agent)

fp.close()

stepcnts = ["500", "1000"]
seeds = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
rewards = {
    (agent, seed, stepcnt): []
    for agent in agents
    for seed in seeds
    for stepcnt in stepcnts
}

for line in lines:
    agent, env, seed, nsteps, reward = line
    scaled_reward = float(reward)/int(nsteps)
    rewards[(agent, seed, nsteps)].append(scaled_reward)

avg_rewards = {}
for agent in agents:
    for seed in seeds:
        for stepcnt in stepcnts:
            s = rewards[(agent, seed, stepcnt)]
            avg_rewards[(agent, seed, stepcnt)] = sum(s)/len(s)

avgs_by_agent_stepcnt = {
    (agent, stepcnt): []
    for agent in agents
    for stepcnt in stepcnts
}

for triple in avg_rewards.keys():
    agent, seed, stepcnt = triple
    avgs_by_agent_stepcnt[(agent, stepcnt)].append(avg_rewards[triple])

for agent in agents:
    for stepcnt in stepcnts:
        avgs = avgs_by_agent_stepcnt[(agent, stepcnt)]
        avg = sum(avgs)/len(avgs)
        var = sum([(x-avg)*(x-avg) for x in avgs])/len(avgs)
        svar = (len(avgs)*var)/(len(avgs)-1)
        sdev = sqrt(svar)
        stderr = sdev/len(avgs)
        print(agent+" ("+stepcnt+" steps): measure "+str(avg)+", stderr: "+str(stderr))
