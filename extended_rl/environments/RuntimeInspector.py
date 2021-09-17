from extended_rl.util import eval_and_count_steps

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