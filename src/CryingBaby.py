FEED, DONTFEED = 0, 1
LAUGH, CRY = 0, 1

class CryingBaby:
    """
    Environment in which the agent must decide when to feed a baby.
    The baby's nutrition level drops over time, but every time the
    agent feeds the baby, the baby's nutrition level rises. Each
    turn, the baby experiences +1 reward if the baby's nutrition
    level is within happy bounds, otherwise, the baby experiences
    -1 reward. Every turn, the baby either cries or laughs, and the
    agent experiences +1 reward if the baby laughs, -1 reward if
    the baby cries. The baby's actions are computed by simulating
    the agent to see what the agent would do in the baby's position.
    """
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 2

    def react(self, T, play):
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
    """
    Compute which action a baby takes in response to a history of
    feedings and non-feedings. This is accomplished by calculating
    a sequence of rewards which the play will have generated for
    the baby: the baby gets +1 reward on each turn when its
    nutrition level is within happy bounds, -1 reward on each turn
    when its nutrition level is not within happy bounds. Along with
    the baby's past cry/laugh actions and observations of being
    fed/not fed, these ingredients are combined into a prompt which
    can be plugged into the parent's action function (T) to
    determine what the parent would do in the baby's position. This
    whole process is sped up using caching techniques.
    """
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