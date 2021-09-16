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
    n_actions = n_obs = 2

    def __init__(self, A):
        self.sim = A()  # Obtain parent by simulating agent
        self.nutrition = 5
        self.prev_action = LAUGH

    def start(self):
        obs = DONTFEED
        return obs

    def step(self, action):
        parent_action = self.sim.act(action)
        if parent_action == FEED:
            self.nutrition = min(self.nutrition+1, 9)
        else:
            self.nutrition = max(self.nutrition-1, 0)

        reward = 1 if (2 < self.nutrition < 8) else -1
        obs = parent_action
        self.sim.train(
            o_prev=self.prev_action,
            a=parent_action,
            r=1 if (action==LAUGH) else -1,
            o_next=action
        )
        self.prev_action = action
        return (reward, obs)