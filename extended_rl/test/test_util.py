# The purpose of this file is to test the library to make sure
# it works. End-users who are not working on contributing code
# to the library do not need to worry about this.


def test_util():
    print("Testing util functions...")
    test_run_environment()
    test_copy_with_meta()
    test_args_to_agent()
    test_prerandoms()

    print("Done testing util functions.")

def test_run_environment():
    from util import run_environment

    num_env_calls = [0]
    num_agent_calls = [0]

    class MockEnv:
        n_actions = n_obs = 1
        def __init__(self, A):
            pass
        def start(self):
            obs = 0
            return obs
        def step(self, action):
            num_env_calls[0] += 1
            reward, obs = 0, 0
            return (reward, obs)

    class MockAgent:
        def __init__(self):
            return
        def act(self, obs):
            num_agent_calls[0] += 1
            return 0
        def train(self, o_prev, a, r, o_next):
            return

    run_environment(MockEnv, MockAgent, 100)
    assert num_env_calls[0] == 100
    assert num_agent_calls[0] == 100

    num_env_calls[0] = 0
    num_agent_calls[0] = 0

    class MockEnv2:
        n_actions = n_obs = 1
        def __init__(self, A):
            self.sim = A()
        def start(self):
            obs = 0
            return obs
        def step(self, action):
            num_env_calls[0] += 1
            self.sim.act(obs=0)
            reward, obs = 0, 0
            return (reward, obs)

    run_environment(MockEnv2, MockAgent, 100)
    assert num_env_calls[0] == 100
    assert num_agent_calls[0] == 200

def test_copy_with_meta():
    from util import copy_with_meta

    class Foo:
        n_actions, n_obs = 99, 5

    class Bar:
        pass

    X = copy_with_meta(Bar, Foo)
    assert X.n_actions == 99
    assert X.n_obs == 5
    assert isinstance(X(), Bar)
    assert X.__name__ == 'Bar'

def test_args_to_agent():
    from util import args_to_agent, run_environment

    class CustomizableConstAgent:
        def __init__(self, action_to_play=0):
            self.action_to_play = action_to_play
        def act(self, obs):
            return self.action_to_play % self.n_actions
        def train(self, o_prev, a, r, o_next):
            pass

    assert CustomizableConstAgent().action_to_play == 0

    class RewardPlaying1:
        n_actions, n_obs = 2, 1
        def __init__(self, A):
            pass
        def start(self):
            return 0
        def step(self, action):
            reward = 1 if (action==1) else -1
            return (reward, 0)

    result = run_environment(RewardPlaying1, CustomizableConstAgent, 10)
    assert result['total_reward'] == -10

    Plays1 = args_to_agent(CustomizableConstAgent, action_to_play=1)
    assert Plays1.__name__ == 'CustomizableConstAgent'
    assert Plays1().kwargs == {'action_to_play': 1}

    result = run_environment(RewardPlaying1, Plays1, 10)
    assert result['total_reward'] == 10

    class RewardHypothetical1:
        n_actions, n_obs = 2, 1
        def __init__(self, A):
            self.sim = A()
        def start(self):
            return 0
        def step(self, action):
            hypothetical_action = self.sim.act(0)
            reward = 1 if (hypothetical_action==1) else -1
            return (reward, 0)

    result = run_environment(RewardHypothetical1, CustomizableConstAgent, 10)
    assert result['total_reward'] == -10

    result = run_environment(RewardHypothetical1, Plays1, 10)
    assert result['total_reward'] == 10

    class OverridesArg:
        n_actions, n_obs = 3, 1
        def __init__(self, A):
            self.sim = A(action_to_play=2)
        def start(self):
            return 0
        def step(self, action):
            assert action == 1
            hypothetical_action = self.sim.act(0)
            reward = 1 if (hypothetical_action==2) else -1
            return (reward, 0)

    result = run_environment(OverridesArg, Plays1, 10)
    assert result['total_reward'] == 10

    # Args are applied to underlying, not to shell.
    # And they are only applied to underlying upon first action.
    class CheckArgs:
        n_actions, n_obs = 2, 1
        def __init__(self, A):
            self.sim = A()
            assert not(hasattr(self.sim, 'action_to_play'))
            assert not(hasattr(self.sim, 'underlying'))
            self.sim.act(0)
            assert self.sim.underlying.action_to_play == 1
        def start(self):
            assert not(hasattr(self.sim, 'action_to_play'))
            return 0
        def step(self, action):
            assert not(hasattr(self.sim, 'action_to_play'))
            return (0,0)

    run_environment(CheckArgs, Plays1, 10)

def test_prerandoms():
    from copy import copy
    import random
    from prerandom import agent_randoms, env_randoms, populate_randoms

    reseed = random.randrange(1_000_000_000)

    agent_randoms_0 = copy(agent_randoms)
    env_randoms_0 = copy(env_randoms)
    populate_randoms()
    assert agent_randoms[0:10] != env_randoms[0:10]
    agent_randoms_1 = copy(agent_randoms)
    env_randoms_1 = copy(env_randoms)
    assert agent_randoms_0[0:10] != agent_randoms_1[0:10]
    assert env_randoms_0[0:10] != env_randoms_1[0:10]
    populate_randoms()
    assert agent_randoms[0:10] != env_randoms[0:10]
    assert agent_randoms_1[0:10] != agent_randoms[0:10]
    assert env_randoms_1[0:100] != env_randoms[0:0]

    d_agent = {}
    d_env = {}

    for seed in range(10):
        populate_randoms(seed)
        assert agent_randoms[0:10] != env_randoms[0:10]
        d_agent[seed] = copy(agent_randoms)
        d_env[seed] = copy(env_randoms)

    for seed in range(10):
        populate_randoms(seed)
        assert agent_randoms == d_agent[seed]
        assert env_randoms == d_env[seed]
        if seed > 0:
            assert agent_randoms[0:10] != d_agent[seed-1][0:10]
            assert env_randoms[0:10] != d_env[seed-1][0:10]

    random.seed(reseed)  # To ensure other tests are non-deterministic