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
    test_naive_learner()
    test_SBL3_agents()

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

def test_naive_learner():
    from agents.naive_learner import NaiveLearner

    total_correct_actions = [0]
    total_actions = [0]

    @annotate(num_legal_actions=5, num_possible_obs=5)
    class EasyEnv:
        def __init__(self, A):
            self.cnt = 0
        def start(self):
            self.prev_obs = self.cnt
            return self.cnt
        def step(self, action):
            if self.cnt >= 1000:
                total_actions[0] += 1
                if action == self.prev_obs:
                    total_correct_actions[0] += 1

            reward = 1 if (action == self.prev_obs) else -1
            self.cnt += 1
            obs = self.cnt % self.num_possible_obs
            self.prev_obs = obs
            return (reward, obs)

    run_environment(EasyEnv, NaiveLearner, 10000)
    assert total_actions[0] == 9000
    percent = total_correct_actions[0] / total_actions[0]
    assert percent > 1 - 2*.15

def test_SBL3_agents():
    try:
        from agents.SBL3_DQN import DQN_learner
        from agents.SBL3_PPO import PPO_learner
        from agents.SBL3_A2C import A2C_learner
    except ModuleNotFoundError:
        print("Skipping test_SBL_agents: dependencies not installed")
        return

    from random import randrange
    from util import args_to_agent

    @annotate(num_possible_obs=2, num_legal_actions=2)
    class EasyEnv:
        def __init__(self, A):
            self.sim=A()
        def start(self):
            obs = randrange(2)
            self.prev_obs = obs
            return obs
        def step(self, action):
            act1 = self.sim.act(self.prev_obs)
            act2 = self.sim.act(self.prev_obs)
            try:
                assert action == act1 == act2
            except Exception:
                import pdb; pdb.set_trace()
            obs = randrange(2)
            reward = 1 if (action == self.prev_obs) else -1
            self.sim.train(
                o_prev=self.prev_obs,
                act=action,
                R=reward,
                o_next = obs
            )
            self.prev_obs = obs
            return (reward, obs)

    for agent in [DQN_learner, A2C_learner, PPO_learner]:
        run_environment(EasyEnv, agent, 100)
