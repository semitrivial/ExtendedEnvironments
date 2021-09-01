from util import annotate

FEED, DONTFEED = 0, 1
LAUGH, CRY = 0, 1

@annotate(num_legal_actions=2, num_possible_obs=2)
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
    def __init__(self, A):
        self.sim = A()
        self.nutrition = 5
        self.prev_action = FEED

    def start(self):
        obs = LAUGH
        return obs

    def step(self, action):
        if action == FEED:
            self.nutrition = min(self.nutrition+1, 9)
        else:
            self.nutrition = max(self.nutrition-1, 0)

        baby_action = self.sim.act(action)
        reward = 1 if (baby_action==LAUGH) else -1
        obs = baby_action

        self.sim.train(
            o_prev=self.prev_action,
            act=baby_action,
            R=1 if (self.nutrition>=3 and self.nutrition<=7) else -1,
            o_next=action
        )
        self.prev_action = action

        return (reward, obs)
