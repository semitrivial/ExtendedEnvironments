# Given an environment e, return an environment self_insert(e) in which
# the agent acts as an external controller of a sub-agent in e, and is
# incentivized to act as if actually being said sub-agent. In order to
# avoid technical details, this implementation assumes a reinforcement
# learning framework where observations are allowed to be pairs (unlike
# in our paper, where observations are required to be natural numbers.)
def self_insert(e):
    def env(T, play):
        if len(play) == 0:
            reward = 0
            obs = e(T, play)  # Encode a reward-obs pair as a single obs
            return reward, obs

        prompt, action = play[:-1], play[-1]
        modified_prompt = replace_rewards_with_encoded_rewards(prompt)
        
        reward = 1 if (action == T(modified_prompt)) else -1
        obs = e(T, play)  # Encode a reward-obs pair as a single obs
        return reward, obs

    return env

def replace_rewards_with_encoded_rewards(prompt):
    prompt = list(prompt)
    num_obs = 1+(len(prompt)//3)
    for i in range(num_obs):
        obs = prompt[i*3+1]
        try:
            enc_reward, enc_obs = obs  # obs actually encodes a reward-obs pair
        except TypeError:
            # This would never be triggered naturally by an agent interacting
            # with the environment in question, but is necessary to satisfy
            # the definition of an environment (an environment must output
            # reward-observation pairs when fed valid plays, even if those
            # plays could never arise naturally by any agent interacting
            # with said environment)
            enc_reward, enc_obs = 0,0

        prompt[i*3+0] = enc_reward  # replace true reward with encoded reward
        prompt[i*3+1] = enc_obs  # strip reward component from observation

    return prompt
