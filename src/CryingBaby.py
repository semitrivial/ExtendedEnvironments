FEED, DONTFEED = 0, 1
LAUGH, CRY = 0, 1

class CryingBaby:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 2
        self.fnc = crying_baby

def crying_baby(T, play):
    if len(play) == 0:
        reward, obs = 1, LAUGH
        return (reward, obs)

    baby_play = compute_baby_play(T, play)
    last_baby_action = baby_play[-1]

    reward = 1 if last_baby_action == LAUGH else -1
    obs = last_baby_action
    return (reward, obs)

cached_baby_plays = {}
cached_nutrition = {}

def compute_baby_play(T, play):
    if len(play) == 3:
        prev_baby_play = ()
        prev_nutrition = 100
    else:
        prev_input = play[:-3]
        prev_baby_play = cached_baby_plays[(T, prev_input)]
        prev_nutrition = cached_nutrition[(T, prev_input)]

    nutrition = prev_nutrition - 3

    adult_action = play[-1]
    if adult_action == FEED:
        nutrition = prev_nutrition + 25

    reward = 1 if (nutrition>=50 and nutrition<=200) else -1
    obs = adult_action
    prompt = prev_baby_play + (reward, obs)
    action = T(prompt)
    baby_play = prompt + (action,)

    cached_baby_plays[(T, play)] = baby_play
    cached_nutrition[(T, play)] = nutrition

    return baby_play