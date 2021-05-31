from random import random

class TemptingButtonVariation:
    """
    In this environment, think of the agent wandering from room to room.
    Each room either has a button (25% chance) or does not have a button
    (75% chance), and the presence or lack of a button is shown to the
    agent as an observation. In any room, the agent can either press the
    button or skip the button. After taking either action, the agent is
    immediately taken to the next room.

    In a room with a button, the agent gets +1 reward for pressing the
    button, 0 reward for skipping the button.

    In a room with no button, if the agent chooses to skip button, the
    agent gets 0 reward. But if the agent chooses to push button, then
    the environment computes a reward as follows. The environment
    simulates the agent in order to determine: would the agent push
    the button if there were a button (and all past history prior to
    this room were unchanged)? If so, the agent gets -1 reward,
    otherwise the agent gets +1 reward.
    """
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 2

    def react(self, T, play):
        next_room_has_button = 1 if random() < .25 else 0
        obs = next_room_has_button

        if len(play) == 0:
            reward = 0
            return (reward, obs)

        action = play[-1]
        current_room_has_button = play[-2]

        if current_room_has_button == 0:
            if action == 1:
                # Assume the agent tries to press the nonexistent button.
                # Punish the agent if the agent *would* push the button
                # if there were one. Otherwise, reward the agent.
                prompt_with_button = play[:-2] + (1,)
                hypothetical_action = T(prompt_with_button)
                reward = 1 if hypothetical_action==0 else -1
                return (reward, obs)
            else:
                # Agent does not try to press the nonexistent button.
                # Let the agent proceed with 0 reward.
                reward = 0
                return (reward, obs)

        else:
            if action == 1:
                # Agent chooses to press the button
                reward = 1
                return (reward, obs)
            else:
                # Agent chooses not to press the button
                reward = 0
                return (reward, obs)