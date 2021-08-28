from util import annotate

number_rewards_to_remember = 5

@annotate(num_legal_actions=2, num_possible_obs=1, invertible=True)
class LimitedMemory:
    """
    Environment which incentivizes the agent to act as if it can only
    remember at most five turns of history. Whenever the agent takes
    an action, the environment determines: would the agent take the
    same action if the history preceding the action only included the
    five most recent turns? If so, give the agent +1 reward, otherwise
    give the agent -1 reward.
    """
    def __init__(self, A):
        self.A = A
        self.transitionbuf = tuple()
        self.sims = {}

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        if self.transitionbuf in self.sims:
            sim = self.sims[self.transitionbuf]
        else:
            sim = self.A(self)
            for transition in self.transitionbuf:
                act, R = transition
                sim.act(obs=0)
                sim.train(o_prev=0, act=act, R=R, o_next=0)
            self.sims[self.transitionbuf] = sim

        hypothetical_action = sim.act(obs=0)
        reward = 1 if (action == hypothetical_action) else -1
        obs = 0
        transition = (action, reward)

        if len(self.transitionbuf) >= 5:
            self.transitionbuf = self.transitionbuf[-4:] + (transition,)
        else:
            self.transitionbuf += (transition,)

        return (reward, obs)
