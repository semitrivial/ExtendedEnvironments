from extended_rl.prerandom import envrandom
from extended_rl.util import annotate

@annotate(n_actions=2, n_obs=8)
class AfterImages:
    """
    An environment where the agent must act as if it sees
    afterimages. The agent is randomly shown "images"
    consisting of 3 bits each (encoded by numbers 0,...,7).
    When the agent acts, the environment determines: is this
    action the same action which the agent would take if each
    image had "bled" into the next image? (For example, if
    010 bleeds into 001, the result would be 011; if 101
    bleeds into 000, the result would be 101; if 011 bleeds
    into 100, the result would be 111; etc.) If the agent
    does take the same action as the agent would take if each
    image had hypothetically bled into the next, then the
    agent is rewarded. Otherwise, the agent is punished.

    Note, each image bleeds into the next, but not into the one
    after the next. So if the first three observations are
    2 (=010), 1 (=001), and 0 (=000),
    then in order to be positively rewarded, the agent must act
    the same way the agent would act if those were
    2 (=010), 3 (=011), and 1 (=001).
    """
    def __init__(self, A):
        self.sim = A()
        self.rnd_counter = 0

    def start(self):
        self.prev_obs = int(envrandom.random(self.rnd_counter) * 8)
        self.prev_bled_obs = self.prev_obs
        return self.prev_obs

    def step(self, action):
        self.rnd_counter += 1
        obs = int(envrandom.random(self.rnd_counter) * 8)
        bled_obs = self.prev_obs | obs
        hypothetical_action = self.sim.act(self.prev_bled_obs)
        reward = 1 if (action == hypothetical_action) else -1

        self.sim.train(
            o_prev=self.prev_bled_obs,
            a=action,
            r=reward,
            o_next=bled_obs
        )

        self.prev_obs = obs
        self.prev_bled_obs = bled_obs
        return (reward, obs)
