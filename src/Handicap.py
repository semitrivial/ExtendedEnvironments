def Handicap(e, h):
    def e_star_h(T, play):
        r_e, o_e = e(T, play)
        r_h, _ = h(T, play)

        if r_h == 1:
            return (r_e, o_e)
        else:
            return (-1, o_e)

    return e_star_h