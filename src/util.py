def run_environment(env, T, num_steps):
    step = 0
    results = {'total_reward': 0.0}
    play = []

    while step < num_steps:
        reward, obs = env(T, play)
        results['total_reward'] += reward
        prompt = play + [reward, obs]
        action = T(prompt)
        play = prompt + [action]
        step += 1

    return results