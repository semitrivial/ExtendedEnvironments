class DelayedRewards:
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
    prompt = list(prompt)
    for i in range(len(prompt)-2, 0, -3):
        if i%6 == 0:
            prompt[i] = prompt[i] + prompt[i-3]
        else:
            prompt[i] = 0

    return tuple(prompt)