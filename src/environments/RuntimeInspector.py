from util import eval_and_count_steps

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
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return reward, obs

        prompt, action = play[:-1], play[1]

        local_vars = {'T': T, 'prompt': prompt}
        _, stepcount = eval_and_count_steps('T(prompt)', local_vars)

        reward = 1 if stepcount < 15*len(prompt) else -1
        obs = 0

        return reward, obs

class PunishFastAgent:
    """
    Environment which punishes agents for taking too few steps to compute
    their actions. Steps are counted by hijacking the Python debugger, pdb.
    When the agent takes an action in response to a prompt of length N,
    simulate the agent on that prompt and give the agent +1 reward if it
    takes the agent >15N pdb steps to compute its output; otherwise, give
    the agent -1 reward. Note, this environment is not included in the
    battery of environments tested against by selfrefl_benchmark, because
    this environment is slow.
    """
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return reward, obs

        prompt, action = play[:-1], play[1]

        local_vars = {'T': T, 'prompt': prompt}
        _, stepcount = eval_and_count_steps('T(prompt)', local_vars)

        reward = 1 if stepcount > 15*len(prompt) else -1
        obs = 0

        return reward, obs
