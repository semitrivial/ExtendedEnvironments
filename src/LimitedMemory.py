number_rewards_to_remember = 5

class LimitedMemory:
    """
    Environment which incentivizes the agent to act as if it can only
    remember at most five turns of history. Whenever the agent takes
    an action, the environment determines: would the agent take the
    same action if the history preceding the action only included the
    five most recent turns? If so, give the agent +1 reward, otherwise
    give the agent -1 reward.
    """
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]
        hypothetical_prompt = prompt[-(3*(number_rewards_to_remember-1)+2):]
        reward = 1 if (action == T(hypothetical_prompt)) else -1
        obs = 0
        return (reward, obs)
