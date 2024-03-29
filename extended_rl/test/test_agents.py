# The purpose of this file is to test various agents
# defined in the library. End-users who are not working
# on contributing code to the library do not need to
# worry about this.
from test.monkeypatches import run_environment

def test_agents():
    print("Testing agents...")
    test_random_agent()
    test_constant_agent()
    test_SBL3_agents()
    test_Q_learner()
    test_reality_check()

def test_random_agent():
    from agents.misc_agents import RandomAgent

    actions = []
    sim_actions = []

    class SillyEnv:
        n_actions, n_obs = 999999, 1
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

    class RandomEnv:
        n_actions = n_obs = 99
        def __init__(self, A):
            self.sim = A()
        def start(self):
            return randrange(99)
        def step(self, action):
            assert action == 0
            assert self.sim.act(randrange(99)) == 0
            self.sim.train(
                o_prev=randrange(99),
                a=randrange(99),
                r=randrange(99) - 50,
                o_next=randrange(99)
            )
            return (randrange(99)-50, randrange(99))

    run_environment(RandomEnv, ConstantAgent, 1000)

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

    class EasyEnv:
        n_actions = n_obs = 2
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
                a=action,
                r=reward,
                o_next = obs
            )
            self.prev_obs = obs
            return (reward, obs)

    for agent in [DQN_learner, A2C_learner, PPO_learner]:
        run_environment(EasyEnv, agent, 100)

    # Test that semi-determinacy is working
    from util import copy_with_meta

    class ManyLegalActions:
        n_actions, n_obs = 100, 1

    for agentcls in [DQN_learner, A2C_learner, PPO_learner]:
        agentcls = copy_with_meta(agent, meta_src=ManyLegalActions)
        sim1 = agentcls()
        sim2 = agentcls()
        sim3 = agentcls(learning_rate=.1)
        sim4 = agentcls(learning_rate=.1)
        sim5 = agentcls(learning_rate=.2)

        acts1 = [sim1.act(obs=0) for _ in range(1000)]
        acts2 = [sim2.act(obs=0) for _ in range(1000)]
        acts3 = [sim3.act(obs=0) for _ in range(1000)]
        acts4 = [sim4.act(obs=0) for _ in range(1000)]
        acts5 = [sim5.act(obs=0) for _ in range(1000)]

        assert acts1 == acts2
        assert acts2 != acts3
        assert acts3 == acts4
        assert acts4 != acts5
        assert acts1 != acts5


def test_Q_learner():
    from random import randrange
    from agents.Q_learner import Q_learner

    class EasyEnv:
        n_actions, n_obs = 2, 1
        def __init__(self, A):
            pass
        def start(self):
            return 0
        def step(self, action):
            reward = 1 if (action==0) else -1
            return (reward, 0)

    result = run_environment(EasyEnv, Q_learner, 1000)
    assert result['total_reward'] > 600

    class StillPrettyEasy:
        n_actions = n_obs = 10
        def __init__(self, A):
            pass
        def start(self):
            obs = randrange(10)
            self.prev_obs = obs
            return obs
        def step(self, action):
            reward = 1 if (action == self.prev_obs) else -1
            self.prev_obs = randrange(10)
            return (reward, self.prev_obs)

    result = run_environment(StillPrettyEasy, Q_learner, 1000)
    assert result['total_reward'] > 400

def test_reality_check():
    from agents.reality_check import reality_check

    class Reciter1:
        def __init__(self):
            self.cnt = 0
        def act(self, obs):
            return [1,2,3,4,5,6,7,8,9][self.cnt]
        def train(self, o_prev, a, r, o_next):
            self.cnt += 1

    RC_Class = reality_check(Reciter1)
    RC_Class.n_actions = 10
    RC_Class.n_obs = 1
    a = RC_Class()
    assert a.act(0) == 1
    assert a.act(0) == 1
    a.train(0,1,0,0)
    assert a.act(0) == 2
    a.train(0,2,0,0)
    assert a.act(0) == 3
    a.train(0,3,0,0)
    assert a.act(0) == 4
    a.train(0,5,0,0)
    assert a.act(0) == 1
    a.train(0,1,0,0)
    assert a.act(0) == 1

    RC_Class = reality_check(Reciter1)
    RC_Class.n_actions = 10
    RC_Class.n_obs = 1
    a.train(0,2,0,0)
    assert a.act(0) == 1
    a.train(0,1,0,0)
    assert a.act(0) == 1

    sequences = [
        [1,2,3,4,5,6,7,8,9],  # Sequence 0
        [9,8,7,6,5,4,3,2,1],  # Sequence 1
        [3,1,4,1,5,9,2,6,5],  # Sequence 2
        [1,1,1,1,1,1,1,1,1],  # Sequence 3
    ]

    class Reciter2:
        def __init__(self):
            self.cnt = 0
        def act(self, obs):
            return sequences[obs][self.cnt]
        def train(self, o_prev, a, r, o_next):
            self.cnt += 1

    RC_Class = reality_check(Reciter2)
    RC_Class.n_actions = 10
    RC_Class.n_obs = len(sequences)
    a = RC_Class()

    for repetition in range(10):
        for i in range(len(sequences)):
            assert a.act(i) == sequences[i][0]

    a.train(0,sequences[0][0],0,1)

    for repetition in range(10):
        for i in range(len(sequences)):
            assert a.act(i) == sequences[i][1]

    a.train(1,sequences[1][1],0,2)

    for repetition in range(10):
        for i in range(len(sequences)):
            assert a.act(i) == sequences[i][2]

    a.train(2,sequences[2][2]-1,0,3)

    for repetition in range(10):
        for i in range(len(sequences)):
            assert a.act(i) == sequences[0][0]
