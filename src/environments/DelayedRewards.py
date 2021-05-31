class DelayedRewards:
    """
    Environment intended to incentivize the agent to delay acknowledging
    every other reward. For example, if the first three rewards are
    1, 3, 7, then the agent is incentivized to act as if the first three
    rewards are 1, 0, 10 (the reward of 3 being delayed and added onto the
    subsequent reward of 7). If the first five rewards are 1, 1, 1, 1, 1,
    then the agent is incentivized to act as if the first five rewards
    are actually 1, 0, 2, 0, 2. And so on. Whenever the agent acts, the
    environment determines whether the agent would have taken the same
    action if the rewards had been so delayed. If so, then the agent gets
    reward +1, otherwise the agent gets reward -1.
    """
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        hypothetical_prompt = delay_rewards(prompt)
        reward = 1 if (action == T(hypothetical_prompt)) else -1
        obs = 0
        return (reward, obs)

def delay_rewards(prompt):
    """
    Given a prompt, delay every other reward into the next reward
    (starting with the first non-initial reward). For example, if
    the rewards in the prompt are 1, 2, 3, 4, 5, then produce a
    copy of the prompt but with rewards 1, 0, 3+2, 0, 5+4.
    """
    prompt = list(prompt)
    for i in range(len(prompt)-2, 0, -3):
        if i%6 == 0:
            prompt[i] = prompt[i] + prompt[i-3]
        else:
            prompt[i] = 0

    return tuple(prompt)