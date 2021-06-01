class ShiftedRewards:
    """
    Environment intended to incentivize the agent to delay acknowledging
    every reward by one turn. For example, if the past rewards were
    1,5,6,2,8, then the agent is incentivized to act as if the rewards
    were 0,1,5,6,2. Whenever the agent takes an action, the environment
    determines whether the agent would have taken the same action if all
    rewards had been so shifted. If so, the agent is given reward +1,
    otherwise the agent is given reward -1.
    """
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        hypothetical_prompt = shift_rewards(prompt)
        reward = 1 if (action == T(hypothetical_prompt)) else -1
        obs = 0
        return (reward, obs)

def shift_rewards(prompt):
    prompt = list(prompt)
    for i in range(len(prompt)-2, 0, -3):
        prompt[i] = prompt[i-3]
    # Note that for any prompt produced by ShiftedRewards,
    # prompt[0] will already be 0 because that is the initial
    # reward in ShiftedRewards. So no need to set prompt[0]=0
    # here.
    return tuple(prompt)