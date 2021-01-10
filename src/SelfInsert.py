def self_insert(base_env):
    def modified_env(T, play):
        if len(play) == 0:
            return base_env(T, play)

        reward0, obs0 = base_env(T, play)
        obs = reverse_hat(reward0, obs0)
        prompt, action = play[:-1], play[-1]
        inner_prompt = decode_observations(prompt)
        reward = 1 if action == T(inner_prompt) else -1
        return [reward, obs]

    return modified_env

def decode_observations(prompt):
    if len(prompt) < 3:
        _, obs = prompt
        return hat(obs)
    else:
        _, obs, action = prompt[0:3]
        return hat(obs) + [action] + decode_observations(prompt[3:])

hat_dict = {}
reverse_hat_dict = {}

def reverse_hat(x, y):
    if not((x,y) in reverse_hat_dict):
        counter = len(reverse_hat_dict)
        reverse_hat_dict[(x,y)] = counter
        hat_dict[counter] = [x,y]

    return reverse_hat_dict[(x,y)]

def hat(n):
    return hat_dict[n]