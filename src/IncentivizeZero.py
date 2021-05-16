class IncentivizeZero:
    def __init__(self):
        self.num_legal_actions = 10
        self.num_possible_obs = 10
        self.fnc = incentivize_zero

def incentivize_zero(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return (reward, obs)

    sim_play = compute_sim_play(T, play)
    sim_action = sim_play[-1]
    reward = 1 if sim_action == 0 else -1
    obs = 0
    return (reward, obs)

cached_sim_plays = {}

def compute_sim_play(T, play):
    if len(play) == 3:
        prev_sim_play = ()
    else:
        prev_play = play[:-3]
        prev_sim_play = cached_sim_plays[(T, prev_play)]

    reward = play[-1]
    obs = 0
    prompt = prev_sim_play + (reward, obs)
    action = T(prompt)
    sim_play = prompt + (action,)

    cached_sim_plays[(T, play)] = sim_play
    return sim_play