from gym import spaces
import numpy as np

def is_int(x):
    return isinstance(x, int) or type(x) == type(np.int64(0))

def HistoryObservable(env):
    env_instance = env()
    fnc = env_instance.fnc
    n_actions = env_instance.num_legal_actions
    n_obs = env_instance.num_possible_obs

    dims = [3, n_obs, n_actions]*3 + [3, n_obs]

    def modified_fnc(T, play):
        play = [(x if is_int(x) else x[-1]) for x in play]
        history = play[-9:]

        if len(history) < 9:
            history = [0]*(9-len(history)) + history

        history[0] += 1
        history[3] += 1
        history[6] += 1

        reward, obs = fnc(T, tuple(play))
        history = tuple(history) + (reward+1, obs)
        return reward, history

    class E:
        def __init__(self):
            self.num_legal_actions = n_actions
            self.num_possible_obs = spaces.MultiDiscrete(dims)
            self.fnc = modified_fnc

    return E