class PunishSlowAgent:
    """
    Environment which punishes agents for taking too many steps to compute
    their actions. Steps are counted by hijacking the Python debugger, pdb.
    When the agent takes an action in response to a prompt of length N,
    simulate the agent on that prompt and give the agent +1 reward if it
    takes the agent <15N pdb steps to compute its output; otherwise, give
    the agent -1 reward. Note, this environment is not included in the
    battery of environments tested against by selfrefl_benchmark, because
    this environment is slow.
    """
    n_actions, n_obs, slow = 2, 1, True

    def __init__(self, A):
        self.sim = A()
        self.turn = 1

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        local_vars = {'sim': self.sim}
        _, stepcount = eval_and_count_steps('sim.act(obs=0)', local_vars)

        reward = 1 if (stepcount < 15*self.turn) else -1
        obs = 0
        self.turn += 1

        self.sim.train(o_prev=0, a=action, r=reward, o_next=0)

        return (reward, obs)

def eval_and_count_steps(str, local_vars):
    # Count how many steps a string of code takes to execute, as measured
    # by the python debugger, pdb. This function works by hijacking pdb.
    # Returns both the result of the underlying code being executed, and
    # the number of steps the execution required.
    stepcount = [0]

    # import pdb here instead of at the top of util.py, so that users who
    # do not use the RuntimeInspector environment will not depend on pdb
    from pdb import Pdb

    # Mock a pdb interface in which the "user" blindly always chooses to
    # "take 1 step" and all outputs from pdb are ignored.
    class consolemock:
        def readline(self):
            stepcount[0] += 1  # Keep track of how many steps go by
            return "s"  # "take 1 step"
        def write(self, *args):
            return
        def flush(self):
            return

    # Execute the given code using the above-mocked interface.
    runner = Pdb(stdin=consolemock(), stdout=consolemock())
    result = runner.runeval(str, locals = local_vars)

    return result, stepcount[0]