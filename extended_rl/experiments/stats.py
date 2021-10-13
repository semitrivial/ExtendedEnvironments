from math import sqrt

# This file is used for taking the individual numbers in result_table.csv
# and turning them into the numbers in the table from Section 6 of
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
    if not(nsteps in stepcnts):
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
  \begin{tabular}{llll}
    \toprule
    Agent & Good Classic & Measure $\pm$ StdErr & Measure $\pm$ StdErr\\
          & Agent? & (Original Agent)     & (Reality Check)\\
    \midrule
""".strip())

results = {}

for agent in agents:
    for stepcnt in stepcnts:
        avgs = avgs_by_agent_stepcnt[(agent, stepcnt)]
        avg = sum(avgs)/len(avgs)
        var = sum([(x-avg)*(x-avg) for x in avgs])/len(avgs)
        svar = (len(avgs)*var)/(len(avgs)-1)
        sdev = sqrt(svar)
        stderr = sdev/len(avgs)

        avg = "{:10.4f}".format(avg).strip()
        stderr = "{:10.4f}".format(stderr).strip()

        if avg == "-0.0000":
            avg = "0.0000"

        if not(agent.startswith("reality_check(")):
            results[agent] = {'original': (avg, stderr)}
        else:
            orig = agent[len("reality_check("):-1]
            results[orig]['rc'] = (avg, stderr)

template = r"""
    {agent} & {goodclassic} & ${orig_measure} \pm {orig_stderr}$ & ${rc_measure} \pm {rc_stderr}$\\
"""[1:-1]

agents_in_order = [
    "RandomAgent",
    "ConstantAgent",
    "SimpleLearner",
    "Q_learner",
    "DQN_learner",
    "A2C_learner",
    "PPO_learner",
]

shortname_dict = {
    "RandomAgent": "Random",
    "ConstantAgent": "Constant",
    "SimpleLearner": "Simple",
    "Q_learner": "Q",
    "DQN_learner": "DQN",
    "A2C_learner": "A2C",
    "PPO_learner": "PPO"
}

is_good_classic = {
    "RandomAgent": "",
    "ConstantAgent": "",
    "SimpleLearner": "",
    "Q_learner": r"$\checkmark$",
    "DQN_learner": r"$\checkmark$",
    "A2C_learner": r"$\checkmark$",
    "PPO_learner": r"$\checkmark$",
}

for agent in agents_in_order:
    result = results[agent]
    orig_measure, orig_stderr = result['original']
    rc_measure, rc_stderr = result['rc']
    print(template.format(
        agent=shortname_dict[agent],
        goodclassic=is_good_classic[agent],
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