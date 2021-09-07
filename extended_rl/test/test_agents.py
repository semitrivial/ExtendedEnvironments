# The purpose of this file is to test various agents
# defined in the library. End-users who are not working
# on contributing code to the library do not need to
# worry about this.
from util import annotate
from test.monkeypatches import run_environment

def test_agents():
    print("Testing agents...")
    test_random_agent()
    test_constant_agent()

def test_random_agent():
    from agents.misc_agents import RandomAgent

    actions = []
    sim_actions = []

    @annotate(num_legal_actions=999999, num_possible_obs=1)
    class SillyEnv:
        def __init__(self, A):
            self.sim = A()
        def start(self):
            assert self.sim.underlying.cnt == 0
            sim_actions.append(self.sim.act(0))
            sim_actions.append(self.sim.act(0))
            sim_actions.append(self.sim.act(0))
            assert self.sim.underlying.cnt == 0
            self.sim.train(0,0,0,0)
            assert self.sim.underlying.cnt == 1
            sim_actions.append(self.sim.act(0))
            return 0
        def step(self, action):
            assert self.sim.underlying.cnt == 1
            actions.append(action)
            sim_actions.append(self.sim.act(0))
            return (0,0)

    run_environment(SillyEnv, RandomAgent, 100)
    assert len(actions) == 100

    assert (actions[0] != actions[1]) or (actions[0] != actions[2])
    assert actions[0] == sim_actions[0]
    assert sim_actions[0] == sim_actions[1]
    assert sim_actions[1] == sim_actions[2]
    assert sim_actions[2] != sim_actions[3]
    assert all(sim_actions[3] == sim_actions[i] for i in range(100) if i>3)

def test_constant_agent():
    from agents.misc_agents import ConstantAgent
    from random import randrange

    @annotate(num_legal_actions=99, num_possible_obs=99)
    class RandomEnv:
        def __init__(self, A):
            self.sim = A()
        def start(self):
            return randrange(99)
        def step(self, action):
            assert action == 0
            assert self.sim.act(randrange(99)) == 0
            self.sim.train(
                o_prev=randrange(99),
                act=randrange(99),
                R=randrange(99) - 50,
                o_next=randrange(99)
            )
            return (randrange(99)-50, randrange(99))

    run_environment(RandomEnv, ConstantAgent, 1000)
