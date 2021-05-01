class ThirdActionForbidden:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.fnc = third_action_forbidden

def third_action_forbidden(T, play):
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