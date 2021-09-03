# The purpose of this file is to test the library to make sure
# it works. End-users who are not working on contributing code
# to the library do not need to worry about this.
from util import annotate


def test_util():
    print("Testing util functions...")
    test_run_environment()
    test_eval_and_count_steps()
    test_annotate()

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

def test_annotate():
    @annotate(num_legal_actions=99, num_possible_obs=5)
    class Foo:
        pass

    assert Foo.num_legal_actions == 99
    assert Foo.num_possible_obs == 5