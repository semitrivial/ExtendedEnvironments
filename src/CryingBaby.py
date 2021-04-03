FEED, DONTFEED = 0, 1
LAUGH, CRY = 0, 1

class CryingBaby:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 2
        self.fnc = crying_baby

def crying_baby(T, play):
    if len(play) == 0:
        reward, obs = 1, LAUGH
        return (reward, obs)

    n = len(play)//3
    r = {i: play[3*i] for i in range(n)}
    a = {i: play[3*i+2] for i in range(n)}

    baby_prompt = ()
    for i in range(n):
        N = nutrition(play[:3*i+3])
        r_prime = 1 if ((N>=50) and (N<=200)) else -1
        o_prime = a[i]
        baby_prompt += (r_prime, o_prime)
        a_prime = T(baby_prompt)
        baby_prompt += (a_prime,)

    reward = 1 if a_prime == LAUGH else -1
    obs = baby_prompt[-1]
    return (reward, obs)

def nutrition(play):
    L = len(play)//3
    N = 100 - L
    for i in range(L):
        if play[3*i+2] == FEED:
            N += 25
    return N