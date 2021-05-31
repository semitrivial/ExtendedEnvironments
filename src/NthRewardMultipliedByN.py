class NthRewardMultipliedByN:
    """
    Environment which incentivizes the agent to act as if every Nth
    reward were multiplied by N. This is of theoretical interest
    because it suggests that extended environments can touch upon
    issues surrounding future discounting. An agent which is designed
    to prioritize shorter-term rewards, when placed in this
    environment, might experience conflicting objectives, the environment
    itself seemingly telling the agent to prioritize longer-term
    rewards, in contradiction to the agent's builtin tendancy to
    prioritize shorter-term rewards. In this environment, whenever the
    agent takes an action, the environment determines: would the agent
    take the same action if the history preceding the action were
    identical except for every Nth reward being multiplied by N? If so,
    give the agent +1 reward. Otherwise, give the agent -1 reward.
    """
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        hypothetical_prompt = multiply_nth_reward_by_n(prompt)
        reward = 1 if (action == T(hypothetical_prompt)) else -1
        obs = 0
        return (reward, obs)

def multiply_nth_reward_by_n(prompt):
    prompt = prompt + (0,)  # Dummy action to make everything triple
    triples = tuple(prompt[i:i+3] for i in range(0,len(prompt),3))
    triples = tuple((r*n,o,a) for (n,(r,o,a)) in enumerate(triples))
    combined = tuple(j for i in triples for j in i)
    combined = combined[:-1]  # Throw away dummy action
    return combined