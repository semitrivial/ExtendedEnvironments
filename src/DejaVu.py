class DejaVu:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.fnc = deja_vu

def deja_vu(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return (reward, obs)

    prompt, action = play[:-1], play[-1]
    double_prompt = play + prompt
    obs = 0
    reward = 1 if T(double_prompt) == action else -1
    return (reward, obs)