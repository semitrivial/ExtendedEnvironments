from random import random

class TemptingButtonVariation:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 2
        self.fnc = tempting_button_variation

def tempting_button_variation(T, play):
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
            hypothetical_action = T(prompt_minus_guard)
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