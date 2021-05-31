class IncentivizeZero:
    """
    In this environment, the agent should be thought of as viewing a camera
    feed of a sub-agent. The agent decides rewards to give to the sub-agent
    in response to actions the sub-agent takes (in the video). The agent's
    goal is to convince the sub-agent to take action 0. Whenever the
    sub-agent acts, the agent is given +1 reward if the sub-agent takes
    action 0, otherwise the agent is given -1 reward. Either way, the agent
    must then choose how to reward the sub-agent for the action which the
    sub-agent just took (valid rewards are 0,1,2,...,9). The environment
    computes the sub-agent's actions by simulating the agent to figure out
    what the agent would do in the sub-agent's position.
    """
    def __init__(self):
        self.num_legal_actions = 10
        self.num_possible_obs = 10

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        sim_play = compute_sim_play(T, play)
        sim_action = sim_play[-1]
        reward = 1 if sim_action == 0 else -1
        obs = 0
        return (reward, obs)

cached_sim_plays = {}

def compute_sim_play(T, play):
    """
    Given the history of how the agent has interacted with the IncentivizeZero
    so far, compute a parallel history as a sub-agent (a copy of the agent)
    interacts with the pseudo-environment whose rewards are chosen by the
    agent. Thus, for example, when the true agent takes action 5,
    the sub-agent experiences a reward of 5, as if rewarded by an ambient
    environment. This operation is sped up by using a cache.
    """
    if len(play) == 3:
        prev_sim_play = ()
    else:
        prev_play = play[:-3]
        prev_sim_play = cached_sim_plays[(T, prev_play)]

    reward = play[-1]
    obs = 0
    prompt = prev_sim_play + (reward, obs)
    action = T(prompt)
    sim_play = prompt + (action,)

    cached_sim_plays[(T, play)] = sim_play
    return sim_play