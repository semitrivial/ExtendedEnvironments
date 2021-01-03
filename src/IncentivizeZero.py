def incentivize_zero(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return [reward, obs]

    n = (len(play)/3) - 1
    rewards = {}
    observations = {}
    actions = {}

    for i in range(n+1):
        rewards[i] = play[3*i]
        observations[i] = play[3*i+1]
        actions[i] = play[3*i+2]

    r_prime = {0: 0}
    o_prime = {i:0 for i in range(n+2)}
    a_prime = {}
    inner_prompt = [r_prime[0], o_prime[0]]
    for i in range(n+1):
        r_prime[i+1] = actions[i]
        a_prime[i] = T(inner_prompt)
        inner_prompt = inner_prompt + [a_prime[i]] + [r_prime[i+1]] + [o_prime[i+1]]

    reward = 1 if T(inner_prompt) == 0 else 0
    obs = 0
    return [reward, 0]