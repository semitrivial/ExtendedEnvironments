def apply_handicap(e, h):
    e_instance, h_instance = e(), h()

    if e_instance.min_reward_per_action < 0:
        raise ValueError("Handicaps only apply to merciful environments")

    def fnc(T, play):
        r_e, o_e = e_instance.fnc(T, play)
        r_h, _ = h_instance.fnc(T, play)

        if r_h == 1:
            return (r_e, o_e)
        else:
            return (-1, o_e)

    n_actions = max(e_instance.num_legal_actions, h_instance.num_legal_actions)

    class E_star_H:
        def __init__(self):
            self.num_legal_actions = n_actions
            self.num_possible_obs = e_instance.num_possible_obs
            self.max_reward_per_action = e_instance.max_reward_per_action
            self.min_reward_per_action = -1
            self.fnc = fnc

    return E_star_H