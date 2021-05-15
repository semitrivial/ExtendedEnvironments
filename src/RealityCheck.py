import numpy as np

def reality_check(T0):
    def T(prompt, num_legal_actions, num_possible_obs):
        for i in range(len(prompt)//3):
            subprompt = prompt[:3*i+2]
            subaction = prompt[3*i+2]
            expected = T0(subprompt, num_legal_actions, num_possible_obs)

            if expected != subaction:
                return T(prompt[:2], num_legal_actions, num_possible_obs)

        return T0(prompt, num_legal_actions, num_possible_obs)

    return T