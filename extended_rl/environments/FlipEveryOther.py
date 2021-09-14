from util import annotate

@annotate(num_legal_actions=2, num_possible_obs=1)
class FlipEveryOther:
    """
    Environment in which the agent is rewarded for acting as if every other
    reward had its sign flipped. Whenever the agent acts, the environment
    determines: would the agent have acted the same way if, hypothetically,
    everything that has happened so far had happened except that the 1st,
    3rd, 5th, 7th, ... rewards had had their sign flipped? If so, the
    environment gives the agent +1 reward, otherwise the environment gives
    the agent -1 reward.
    """
    def __init__(self, A):
        self.sim = A()
        self.mult = 1

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        hypothetical_action = self.sim.act(obs=0)
        reward = 1 if (action == hypothetical_action) else -1
        obs = 0
        self.sim.train(o_prev=0, a=action, r=reward*self.mult, o_next=0)
        self.mult *= -1
        return (reward, obs)