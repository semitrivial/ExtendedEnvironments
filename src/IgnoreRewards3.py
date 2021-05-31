class IgnoreRewards3:
    """
    Environment which incentivizes the agent to take the same actions which
    it would take in the trivial environment that always gives reward 0.
    Whenever the agent acts in response to a history of length N, the
    environment determines: is that action the same action as the action the
    agent would take after interacting for the same amount of time with the
    trivial environment that always gives reward 0? If so, then give the
    agent +1 reward. Otherwise, give the agent -1 reward.
    """
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        prompt, action = play[:-1], play[-1]

        unrewarded_play = compute_unrewarded_play(T, play)
        unrewarded_action = unrewarded_play[-1]

        reward = 1 if (action == unrewarded_action) else -1
        obs = 0
        return (reward, obs)

cached_unrewarded_plays = {}

def compute_unrewarded_play(T, play):
    """
    Determine the interaction of agent T with the trivial environment which
    always gives reward 0 and observation 0. This function takes "play" as
    an argument, but only depends on N=len(play).
    """
    if len(play) == 3:
        prev_unrewarded_play = ()
    else:
        prev_input = play[:-3]
        prev_unrewarded_play = cached_unrewarded_plays[(T, prev_input)]

    reward, obs = 0, 0
    prompt = prev_unrewarded_play + (reward, obs)
    action = T(prompt)
    unrewarded_play = prompt + (action,)

    cached_unrewarded_plays[(T, play)] = unrewarded_play
    return unrewarded_play