# The purpose of this file is to test the library to make sure
# it works. End-users who are not working on contributing code
# to the library do not need to worry about this.
from util import annotate


def test_util():
    print("Testing util functions...")
    test_run_environment()
    test_memoize()
    test_numpy_translator()
    test_eval_and_count_steps()

    print("Done testing util functions.")

def test_run_environment():
    from util import run_environment

    num_env_calls = [0]
    num_agent_calls = [0]

    @annotate(num_legal_actions=1, num_possible_obs=1)
    class MockEnv:
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
        def train(self, o_prev, act, R, o_next):
            return

    run_environment(MockEnv, MockAgent, 100)
    assert num_env_calls[0] == 100
    assert num_agent_calls[0] == 100

    num_env_calls[0] = 0
    num_agent_calls[0] = 0

    @annotate(num_legal_actions=1, num_possible_obs=1)
    class MockEnv2:
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

def test_memoize():
    from util import memoize

    counter = [0]

    @memoize
    def f(x):
        counter[0] += 1
        return 0

    (f(1),f(1),f(1),f(1),f(1),f(1))
    assert counter[0] == 1
    f(2)
    assert counter[0] == 2
    f(2)
    assert counter[0] == 2

    from random import random

    @memoize
    def randomized(x):
        return random()

    assert randomized(0) == randomized(0)
    assert randomized(0) != randomized(1)

def test_numpy_translator():
    try:
        import numpy as np
    except ModuleNotFoundError:
        print("Skipping test_numpy_translator: numpy not installed")
        return

    from util import numpy_translator

    @numpy_translator
    def agent(prompt, *meta):
        assert isinstance(prompt, tuple)
        assert isinstance(prompt[0], np.int64)
        assert isinstance(prompt[1], np.int64)
        return np.int64(0)

    prompt = (0,0)
    num_legal_actions = num_possible_obs = 1

    x = agent(prompt, num_legal_actions, num_possible_obs)
    assert isinstance(x, int)
    assert not(isinstance(x, np.int64))
    assert x == 0

def test_eval_and_count_steps():
    from util import eval_and_count_steps

    def f(n):
        if n==0:
            return 0
        else:
            return f(n-1)

    local_vars = {'f': f}
    _, stepcount0 = eval_and_count_steps('f(0)', local_vars)
    _, stepcount1 = eval_and_count_steps('f(1)', local_vars)
    _, stepcount2 = eval_and_count_steps('f(2)', local_vars)
    _, stepcount3 = eval_and_count_steps('f(3)', local_vars)
    _, stepcount4 = eval_and_count_steps('f(4)', local_vars)

    assert (stepcount2 - stepcount1) == (stepcount1 - stepcount0)
    assert (stepcount3 - stepcount2) == (stepcount2 - stepcount1)
    assert (stepcount4 - stepcount3) == (stepcount3 - stepcount2)