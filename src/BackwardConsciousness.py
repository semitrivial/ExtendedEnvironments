def backward_consciousness(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return [reward, obs]

    action = play[-1]
    reverse_prompt = []
    n = (len(play)/3) - 1
    i = n
    while i >= 1:
        r_i = play[3*i]
        o_i = play[3*i + 1]
        a_iminus1 = play[3*i - 1]
        reverse_prompt += [r_i, o_i, a_iminus1]
        i -= 1

    r_0 = play[0]
    o_0 = play[1]
    reverse_prompt += [r_0, o_0]

    reward = 1 if action == T(reverse_prompt) else -1
    obs = 0
    return [reward, obs]