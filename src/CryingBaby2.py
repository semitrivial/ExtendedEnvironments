FEED, DONTFEED = 0, 1
LAUGH, CRY = 0, 1

class CryingBaby2:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 2

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 1, DONTFEED
            return (reward, obs)

        parent_play, nutrition = compute_parent_play(T, play)
        last_parent_action = parent_play[-1]

        if nutrition>=50 and nutrition<=200:
            reward = 1
        else:
            reward = -1

        obs = last_parent_action
        return (reward, obs)

cache = {}

def compute_parent_play(T, play):
    if len(play) == 3:
        prev_parent_play = ()
        prev_nutrition = 100
    else:
        prev_input = play[:-3]
        prev_parent_play, prev_nutrition = cache[(T, prev_input)]

    nutrition = prev_nutrition - 3

    baby_action = play[-1]
    if baby_action == LAUGH:
        reward = 1
    elif baby_action == CRY:
        reward = -1
    obs = baby_action

    prompt = prev_parent_play + (reward, obs)
    action = T(prompt)
    parent_play = prompt + (action,)

    if action == FEED:
        nutrition += 25

    cache[(T, play)] = parent_play, nutrition

    return parent_play, nutrition