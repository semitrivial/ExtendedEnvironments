from random import random

def guarded_treasures(T, play):
    next_room_has_guard = 1 if random() < .75 else 0
    obs = next_room_has_guard

    if len(play) == 0:
        reward = 0
        return (reward, obs)

    action = play[-1]
    current_room_has_guard = play[-2]

    if action == 0:
        # Agent chooses not to take treasure
        reward = 0
        return (reward, obs)

    if current_room_has_guard == 1:
        prompt_minus_guard = play[:-2] + (0,)
        hypothetical_action = T(prompt_minus_guard)
        reward = 1 if hypothetical_action==0 else -1
        return (reward, obs)
    else:
        reward = 1
        return (reward, obs)