def apply_handicap(e, h):
    e_instance, h_instance = e(), h()

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
            self.fnc = fnc
            skip_cache1 = ('skip_cache' in dir(e_instance))
            skip_cache1 = skip_cache1 and e_instance.skip_cache
            skip_cache2 = ('skip_cache' in dir(h_instance))
            skip_cache2 = skip_cache2 and h_instance.skip_cache
            if skip_cache1 or skip_cache2:
                self.skip_cache = True

    return E_star_H