# Extended Environments

An "Extended Environment" is a reinforcement learning environment which is aware of
the source-code of whatever agent is interacting with it. This enables the
environment to simulate the agent and use the results of such simulations as part
of how it determines which rewards and observations to send to the agent. Although
this is a departure from traditional RL environments (which are not able to simulate
agents), nevertheless, a traditional RL agent does not require any extension in
order to interact with an Extended Environment. Thus, Extended Environments can be
used to benchmark RL agents in ways that traditional RL environments cannot.

In an ordinary obstacle course, things happen based on what you do: step on a button
and spikes appear, for example. Imagine an obstacle course where things happen
based on what you would hypothetically do: enter a room with no button and spikes
appear if you would step on the button if there hypothetically were one. Such an
environment would be impossible to stage for a human participant because it is
impossible to determine what a human would hypothetically do in some counterfactual
scenario. But if instead of a human we allow an AI participant to enter the
environment, then the environment can indeed determine what the AI participant would
do in hypothetical scenarios, provided the environment is made aware of the AI's
source-code.

By basing rewards on what the agent would hypothetically do, it is possible for an
Extended Environment to incentivize tasks that seemingly require some degree of
self-awareness. For example, the environment can reward the agent for acting exactly
as the agent would act if the agent was always given reward 0. If the agent does not
so act, the environment can punish the agent. This is possible because the environment
is able to give the agent all-0 rewards in a simulation in order to determine what
action the agent would take in response. Thus, paradoxically, an Extended Environment
can "reward the agent for ignoring rewards". This incentivizes the agent to
self-reflect and ask itself: "Although in reality I have been given positive and
negative rewards in this environment, what action would I take if I had instead always
received zero reward?" This seems to require a degree of self-awareness.

We hope that by running various RL agents against a battery of Extended Environments
that incentivize various types of self-awareness, it will be possible to empirically
measure to what degree said agent is self-aware.

# Roadmap

This Library of Extended Environments is still in very active development and has
not yet been announced to the world. At this stage, it is probably still a ways away
from what it will ultimately look like. In short, this library is Under Construction.