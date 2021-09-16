from extended_rl.prerandom import envrandom

class IgnoreObservations:
    """
    Environment which incentivizes the agent to ignore past observations.
    Whenever the agent acts, the environment determines: would the agent
    have acted the same if the history preceding the action were identical
    except for all observations being 0? If so, then give the agent +1
    reward. Otherwise, give the agent -1 reward.
    """
    n_actions, n_obs = 2, 5

    def __init__(self, A):
        self.sim = A()
        self.rnd_counter = 0

    def start(self):
        self.prev_obs = envrandom.randrange(5, self.rnd_counter)
        return self.prev_obs

    def step(self, action):
        hypothetical_action = self.sim.act(obs=self.prev_obs)
        reward = 1 if (action == hypothetical_action) else -1
        self.rnd_counter += 1
        obs = envrandom.randrange(5, self.rnd_counter)
        self.sim.train(o_prev=0, a=action, r=reward, o_next=0)
        return (reward, obs)