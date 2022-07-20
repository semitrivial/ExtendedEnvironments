<img align="right" width="201" height="195" src="logo.png">

This library is a collection of some extended environments. Extended environments are defined in our paper, "Extending Environments to Measure Self-reflection in Reinforcement Learning".

# Extended Environments

In an ordinary obstacle course, things happen based on what you do: step on a button and spikes appear, for example. Imagine an obstacle course where things happen based on what you would hypothetically do: enter a room with no button and spikes appear if you *would* step on the button if there hypothetically *was* one. Such an environment would be impossible to stage for a human participant, because it is impossible to determine what a human would hypothetically do in some counterfactual scenario. But if we have the source-code of an AI participant, then we **can** determine what that participant would do in hypothetical scenarios, and so we **can** put AI participants into such obstacle courses.

An *extended environment* is a reinforcement learning environment which is able to simulate the agent and use the results when it determines which rewards and observations to send to the agent. Although this is a departure from traditional RL environments (which are not able to simulate agents), nevertheless, a traditional RL agent does not require any extension in order to interact with an Extended Environment.

If an agent does not self-reflect about its own actions, then an extended environment might be difficult for the agent to figure out. Therefore, our thesis is that self-reflection is needed for an agent to achieve good performance on average over the space of all extended environments.

## Layout of this library

* extended_rl/environments: Example extended environments

* extended_rl/agents: Agents (including versions of StableBaselines3 agents adapted to be able to run in extended environments)

* extended_rl/experiments: Scripts used to run experiments used to generate numerical data for our paper
