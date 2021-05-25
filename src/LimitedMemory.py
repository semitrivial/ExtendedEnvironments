number_rewards_to_remember = 5

class LimitedMemory:
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
