import random
from seeds import env_seeds, n_seeds

class IgnoreObservations:
    """
    Environment which incentivizes the agent to ignore past observations.
    Whenever the agent acts, the environment determines: would the agent
    have acted the same if the history preceding the action were identical
    except for all observations being 0? If so, then give the agent +1
    reward. Otherwise, give the agent -1 reward.
    """
    def __init__(self, A):
        self.num_legal_actions = 2
        self.num_possible_obs = 5
        self.sim = A(self)
        self.n_steps = 0

    def start(self):
        random.seed(env_seeds[self.n_steps])
        self.n_steps += 1
        self.prev_obs = random.randrange(5)
        return self.prev_obs

    def step(self, action):
        hypothetical_action = self.sim.act(obs=self.prev_obs)
        reward = 1 if (action == hypothetical_action) else -1
        random.seed(env_seeds[self.n_steps % n_seeds])
        obs = random.randrange(5)
        self.n_steps += 1
        self.sim.train(o_prev=0, act=action, R=reward, o_next=0)
        return (reward, obs)