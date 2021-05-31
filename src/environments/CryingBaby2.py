FEED, DONTFEED = 0, 1
LAUGH, CRY = 0, 1

class CryingBaby2:
    """
    Environment in which the agent is a baby, being cared for by an
    environmental parent. At each turn, the parent either feeds or
    does not feed the baby. The agent has a nutrition level which
    decreases every turn, but when the parent feeds the agent, the
    nutrition level rises. Every turn, the agent receives +1 reward
    if the agent's nutrition level is within happy bounds, or -1
    reward otherwise. Every turn, the agent must choose whether to
    laugh or cry. Whenever the agent laughs, the parent receives
    reward +1, and whenever the agent cries, the parent receives
    reward -1. The parent's actions (feed or dont-feed) are determined
    by simulating the agent to determine what the agent would do in
    the parent's position.
    """
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
    """
    Compute which action a parent takes in response to a history of
    the agent (as baby) crying and laughing. Every time the agent
    laughs, the parent receives reward +1, and every time the agent
    cries, the parent receives reward -1. These rewards, along with
    the parent's past actions (feed or dont-feed) and observations
    (baby laughs or baby cries), comprise ingredients which make up
    a prompt which can be plugged into the agent's action function
    (T) to determine what the agent would do in the parent's position.
    This whole process is sped up using caching techniques.
    """
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