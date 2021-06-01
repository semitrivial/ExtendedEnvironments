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

import pdb; pdb.set_trace()
print("...")