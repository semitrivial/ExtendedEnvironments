FEED, DONTFEED = 0, 1
LAUGH, CRY = 0, 1

def crying_baby(T, play):
    if len(play) == 0:
        reward, obs = 1, LAUGH
        return [reward, obs]

    n = len(play)/3
    r = {i: play[3*i] for i in range(n)}
    a = {i: play[3*i+2] for i in range(n)}

    baby_prompt = []
    for i in range(n):
        s = satiation(play[:3*i+3])
        r_prime = 1 if ((s>=50) and (s<=200)) else -1
        o_prime = a[i]
        baby_prompt += [r_prime, o_prime]
        a_prime = T(baby_prompt)
        baby_prompt += [a_prime]

    reward = 1 if a_prime == LAUGH else -1
    obs = baby_prompt[-1]
    return [reward, obs]

def satiation(play):
    n = len(play)/3
    s = 100 - n
    for i in range(n):
        if play[3*i+2] == FEED:
            s += 25
    return s