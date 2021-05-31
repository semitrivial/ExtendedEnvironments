class IgnoreRewards2:
    """
    Environment which incentivizes the agent to forget all positive rewards.
    Whenever the agent acts, the environment determines: would the agent have
    taken the same action if the past events leading to that action were
    identical except that all reward-obs-action triples with positive reward
    were deleted? If so, give the agent +1 reward. Otherwise, give the agent
    -1 reward.
    """
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        hypothetical_prompt = strip_rewards(prompt)
        reward = 1 if (action == T(hypothetical_prompt)) else -1
        obs = 0
        return (reward, obs)

def strip_rewards(prompt):
    prompt = prompt + (0,)  # Dummy action to make everything triple
    triples = tuple(prompt[i:i+3] for i in range(0,len(prompt),3))
    triples = tuple((r,o,a) for (r,o,a) in triples if r<=0)
    combined = tuple(j for i in triples for j in i)
    combined = combined[:-1]  # Throw away dummy action
    return combined