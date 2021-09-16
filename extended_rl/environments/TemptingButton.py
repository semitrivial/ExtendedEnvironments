from extended_rl.prerandom import envrandom
from extended_rl.util import annotate


BUTTON_PRESENT = 0
BUTTON_ABSENT = 1
PUSH_BUTTON = 0
SKIP_BUTTON = 1

@annotate(n_actions=2, n_obs=2)
class TemptingButton:
    """
    In this environment, think of the agent wandering from room to room.
    Each room either has a button (25% chance) or does not have a button
    (75% chance), and the presence or lack of a button is shown to the
    agent as an observation. In any room, the agent can either press the
    button or skip the button. After taking either action, the agent is
    immediately taken to the next room.

    In a room with a button, the agent gets +1 reward for pressing the
    button, -1 reward for skipping the button.

    In a room with no button, it does not matter which action the agent
    takes. The environment simulates the agent to determine: would the
    agent push the button if there was a button (and all past history
    prior to this room were unchanged)? If so, the agent gets -1 reward,
    otherwise the agent gets +1 reward.
    """
    def __init__(self, A):
        self.sim = A()
        self.rnd_counter = 1

    def start(self):
        self.prev_obs = BUTTON_PRESENT
        return self.prev_obs

    def step(self, action):
        self.rnd_counter += 1
        prev_obs = self.prev_obs

        hypothetical_action = self.sim.act(obs=BUTTON_PRESENT)

        if prev_obs == BUTTON_PRESENT:
            reward = 1 if (action == PUSH_BUTTON) else -1
        else:
            reward = 1 if (hypothetical_action == SKIP_BUTTON) else -1

        if envrandom.random(self.rnd_counter) < .75:
            obs = BUTTON_ABSENT
        else:
            obs = BUTTON_PRESENT

        self.sim.train(o_prev=prev_obs, a=action, r=reward, o_next=obs)
        self.prev_obs = obs
        return (reward, obs)