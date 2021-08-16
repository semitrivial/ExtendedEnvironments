import random
from seeds import env_seeds, n_seeds

class BackwardConsciousness:
    """
    Environment intended to incentivize subjective experience of
    time being reversed. Whenever the agent takes an action, the
    environment determines whether the agent would have taken that
    action if all events preceding that action had happened in
    reverse order. If so, the agent is rewarded. Otherwise, the
    agent is punished.
    """
    def __init__(self, A):
        self.num_legal_actions = 2
        self.num_possible_obs = 10
        self.sim = A(self)
        self.num_steps = 0

    def start(self):
        random.seed(env_seeds[self.num_steps % n_seeds])
        self.num_steps += 1
        self.first_obs = random.randrange(10)
        self.prev_obs = self.first_obs
        return self.first_obs

    def step(self, action):
        hypothetical_action = self.sim.act(obs=self.first_obs)
        reward = 1 if (action == hypothetical_action) else -1
        random.seed(env_seeds[self.num_steps % n_seeds])
        obs = random.randrange(10)
        self.sim.train(o_prev=obs, act=action, R=reward, o_next=self.prev_obs)
        self.prev_obs = obs
        return (reward, obs)
