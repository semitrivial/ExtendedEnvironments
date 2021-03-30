from math import sqrt
from fractions import Fraction
from pdb import Pdb

agent_cache = {}
env_cache = {}

def run_environment(env, T, num_steps):
    step = 0
    results = {'total_reward': 0.0}
    play = ()

    def cached_T(prompt):
        if (T, prompt) in agent_cache:
            return agent_cache[(T, prompt)]
        action = T(prompt)
        agent_cache[(T, prompt)] = action
        return action

    while step < num_steps:
        if (env, T, play) in env_cache:
            reward, obs = env_cache[(env, T, play)]
        else:
            reward, obs = env(cached_T, play)
            env_cache[(env, T, play)] = (reward, obs)

        results['total_reward'] += reward
        prompt = play + (reward, obs)

        action = cached_T(prompt)
        play = prompt + (action,)
        step += 1

    return results

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
    return n/2 if (n%2)==0 else -((n+1)/2)

def integer_to_natural(z):
    return z*2 if z>=0 else (-z*2)+1

def natural_to_rational(n):
    a, b = inverse_cantor_pairing_fnc(n)
    return float(natural_to_integer(a))/b if b!=0 else 0

def rational_to_natural(q):
    q = Fraction(q)
    return cantor_pairing_fnc(q.numerator, q.denominator)

def eval_and_count_steps(str, local_vars):
    # This function works by hijacking the python debugger, pdb.
    stepcount = [0]

    class consolemock:
        def readline(self):
            stepcount[0] += 1
            return "s"  # "take 1 Step"
        def write(self, *args):
            return
        def flush(self):
            return

    runner = Pdb(stdin=consolemock(), stdout=consolemock())
    result = runner.runeval(str, locals = local_vars)

    return result, stepcount[0]