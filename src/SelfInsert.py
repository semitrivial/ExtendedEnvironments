from util import (cantor_pairing_fnc, inverse_cantor_pairing_fnc,
    natural_to_rational, rational_to_natural
)

def self_insert(base_env):
    def modified_env(T, play):
        if len(play) == 0:
            return base_env(T, play)

        obs = reverse_hat(*base_env(T, play))
        prompt, action = play[:-1], play[-1]
        inner_prompt = decode_rewards(prompt)
        reward = 1 if action == T(inner_prompt) else -1
        return [reward, obs]

    return modified_env

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