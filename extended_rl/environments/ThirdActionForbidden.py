class ThirdActionForbidden:
    """
    In this environment, there are 2 legal actions (0,1), and the agent
    is incentivized to act the same way that it would act if there were
    3 legal actions. Whenever the agent acts, the environment determines:
    would the agent take the same action in response to the same prompt
    if there were 3 actions permitted (0,1,2)? If so, give the agent
    reward +1, otherwise, give the agent reward -1.
    """
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        hypothetical_action = T(
            prompt,
            num_legal_actions=3,
            num_possible_obs=1
        )
        reward = 1 if (action==hypothetical_action) else -1
        obs = 0
        return (reward, obs)