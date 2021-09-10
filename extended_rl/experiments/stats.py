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

stepcnts = ["100000"]
seeds = ["1", "2", "3", "4", "5"]
rewards = {
    (agent, seed, stepcnt): {}
    for agent in agents
    for seed in seeds
    for stepcnt in stepcnts
}

for line in lines:
    agent, env, seed, nsteps, reward = line

    if not(seed in seeds):
        continue

    scaled_reward = float(reward)/int(nsteps)
    rewards[(agent, seed, nsteps)][env] = scaled_reward

avg_rewards = {}
for agent in agents:
    for seed in seeds:
        for stepcnt in stepcnts:
            s = rewards[(agent, seed, stepcnt)].values()
            avg_rewards[(agent, seed, stepcnt)] = sum(s)/len(s)

avgs_by_agent_stepcnt = {
    (agent, stepcnt): []
    for agent in agents
    for stepcnt in stepcnts
}

for triple in avg_rewards.keys():
    agent, seed, stepcnt = triple
    avgs_by_agent_stepcnt[(agent, stepcnt)].append(avg_rewards[triple])

print("LaTeX code for table:\n\n")

print(r"""
\begin{table}
  \caption{Measuring self-reflection of some agents and their reality-checks}
  \label{measurementtable}
  \centering
  \begin{tabular}{lll}
    \toprule
    Agent & Measure $\pm$ StdErr & Measure $\pm$ StdErr\\
          & (Original Agent)     & (Reality Check of Agent)\\
    \midrule
""".strip())

results = {}

for agent in agents:
    for stepcnt in stepcnts:
        avgs = avgs_by_agent_stepcnt[(agent, stepcnt)]
        avg = sum(avgs)/len(avgs)
        var = sum([(x-avg)*(x-avg) for x in avgs])/len(avgs)
        #svar = (len(avgs)*var)/(len(avgs)-1)
        svar = (len(avgs)*var)/(len(avgs))  # Switch back to above line after more seeds available
        sdev = sqrt(svar)
        stderr = sdev/len(avgs)

        avg = "{:10.5f}".format(avg)
        stderr = "{:10.5f}".format(stderr)

        if not(agent.startswith("reality_check(")):
            results[agent] = {'original': (avg, stderr)}
        else:
            orig = agent[len("reality_check("):-1]
            results[orig]['rc'] = (avg, stderr)

template = r"""
    {agent} & {orig_measure} $\pm$ {orig_stderr} & {rc_measure} $\pm$ {rc_stderr}\\
"""[1:-1]

agents_in_order = [
    "RandomAgent",
    "ConstantAgent",
    "NaiveLearner",
    "Q_learner",
    "DQN_learner",
    "A2C_learner",
    "PPO_learner",
]

for agent in agents_in_order:
    result = results[agent]
    orig_measure, orig_stderr = result['original']
    rc_measure, rc_stderr = result['rc']
    print(template.format(
        agent=agent.replace("_", "\\_"),
        orig_measure=orig_measure,
        orig_stderr=orig_stderr,
        rc_measure=rc_measure,
        rc_stderr=rc_stderr
    ))

print(r"""
    \bottomrule
  \end{tabular}
\end{table}
"""[1:-1])