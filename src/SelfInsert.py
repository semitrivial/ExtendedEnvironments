from math import sqrt
from fractions import Fraction

def self_insert(base_env):
    def modified_env(T, play):
        if len(play) == 0:
            return base_env(T, play)

        obs = reverse_hat(base_env(T, play))
        prompt, action = play[:-1], play[-1]
        inner_prompt = decode_rewards(prompt)
        reward = 1 if action = T(inner_prompt) else -1
        return [reward, obs]

def decode_rewards(prompt):
    if len(prompt) < 3:
        _, obs = prompt
        return hat(obs)
    else:
        _, obs, action = prompt[0:3]
        return hat(obs) + [action] + decode_rewards(prompt[3:])

def hat(obs):
    a, b = inverse_cantor_pairing_fnc(obs)
    return [natural_to_rational(a), b]

def reverse_hat(reward, obs):
    return cantor_pairing_fnc(rational_to_natural(reward), obs)

def cantor_pairing_fnc(k1,k2):
    # From https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    # As accessed on Jan 3rd, 2021
    return (.5*(k1+k2)*(k1+k2+1)) + k2

def inverse_cantor_pairing_fnc(z):
    # From https://en.wikipedia.org/wiki/Pairing_function#Cantor_pairing_function
    # As accessed on Jan 3rd, 2021
    w = int((sqrt(8*z+1)-1)*.5)
    t = (w*w+w)/2
    y = z-t
    x = w-y
    return [x,y]

def natural_to_integer(n):
    return n/2 if (n%2)==0 else -((n-1)/2)

def integer_to_natural(z):
    return z*2 if z>=0 else (-z*2)+1

def natural_to_rational(n):
    a, b = inverse_cantor_pairing_fnc(n)
    return float(natural_to_integer(a)) / b

def rational_to_natural(q):
    q = Fraction(q)
    return cantor_pairing_fnc(q.numerator, q.denominator)