def test_util():
    print("Testing util functions...")
    test_run_environment()

    print("Done testing util functions.")

def test_run_environment():
    from util import run_environment

    num_env_calls = [0]
    num_agent_calls = [0]

    class MockEnv:
        def __init__(self):
            self.num_legal_actions = 1
            self.num_possible_obs = 1
        def react(self, T, play):
            num_env_calls[0] += 1
            reward, obs = 0, 0
            return (reward, obs)

    def mock_agent(prompt, num_legal_actions, num_possible_obs, **kwargs):
        num_agent_calls[0] += 1
        return 0

    run_environment(MockEnv, mock_agent, 100)
    assert num_env_calls[0] == 100
    assert num_agent_calls[0] == 100

    num_env_calls[0] = 0
    num_agent_calls[0] = 0

    class MockEnv2:
        def __init__(self):
            self.num_legal_actions = 1
            self.num_possible_obs = 1
        def react(self, T, play):
            num_env_calls[0] += 1
            T((0,0))
            return (0,0)

    run_environment(MockEnv2, mock_agent, 100)
    assert num_env_calls[0] == 100
    assert num_agent_calls[0] == 200