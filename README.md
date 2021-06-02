# Extended Environments

Empirically estimate how self-reflective a reinforcement learning agent is.
This proof-of-concept library contains 25 so-called "extended environments"
and infrastructure allowing you to run a reinforcement learning agent against
those environments. Figuring out these environments seems to require that an
agent self-reflect about itself (see Theory below), therefore an agent's
performance in these environments can serve as a rough empirical estimate of
how self-reflective the agent is.

**Note:** As this library is first-of-kind, we have made no attempt to
optimize it. It is meant to serve more as a proof-of-concept. Rather than
strenuously optimize the environments in the library, we have instead
designed environments of theoretical interest. Measurements obtained from
this library should not be used to make real-world policy-decisions related
to self-reflection. Self-reflection should not be confused with consciousness
(the two might be related, but that is beyond the scope of this library).


## Theory

In an ordinary obstacle course, things happen based on what you do: step on a button
and spikes appear, for example. Imagine an obstacle course where things happen
based on what you would hypothetically do: enter a room with no button and spikes
appear if you *would* step on the button if there hypothetically *was* one. Such an
environment would be impossible to stage for a human participant because it is
impossible to determine what a human would hypothetically do in some counterfactual
scenario. But if we have the source-code of an AI participant, then we **can**
determine what that participant would do in hypothetical situations, and so we
**can** put AI participants into such obstacle courses.

An "Extended Environment" is a reinforcement learning environment which is aware of
the source-code of whatever agent is interacting with it. This enables the
environment to simulate the agent and use the results when it determines which
rewards and observations to send to the agent. Although this is a departure from
traditional RL environments (which are not able to simulate agents), nevertheless,
a traditional RL agent does not require any extension in order to interact with an
Extended Environment. Thus, Extended Environments can be used to benchmark RL agents
in ways that traditional RL environments cannot.

If an agent does not self-reflect about its own actions, then an extended
environment might be difficult for the agent figure out how the environment
works. Therefore, our thesis is that self-reflection is needed for an agent to
achieve good performance averaged over a battery of suitably chosen extended
environments. This would imply that by measuring how an agent performs across
such a battery of environments, it is possible to empirically estimate how
self-reflective an agent is.

## Installation

**Note:** The library has been built and tested using Python 3.6, so
we recommend using that version or later of python for running the library.

### Install using pip

Just like all other python packages, we recommend installing
Extended Environments in a virtualenv or a conda environment.

To install, `cd` into the cloned repository and do a local pip install:
```
cd ExtendedEnvironments
pip install -e .
```

Optionally, if you wish to use the Stable Baselines3 agents in
`agents/SBL3_agents.py` (required for running the experiment in the
`experiments` directory) you will additionally need to install
Stable Baselines3:
```
pip install stable-baselines3
```

