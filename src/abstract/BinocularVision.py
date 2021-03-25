from util import cantor_pairing_fnc

# Take a 3d game and two camera functions and output an extended environment
# designed to incentivize binocular vision.
# -------------
# Inputs:
#   Game3D: a function which outputs 3d models as a function of actionsequences.
#     A 3d model is, e.g., a 100x100x100 matrix of 1s and 0s, representing the 3d
#     world in front of the player.
#     An actionsequence is a sequence of actions taken by the player.
#   LeftCamera: a function which takes a 3d model and outputs a 2d bitmap of
#     what the player's left eye sees of that 3d model
#   RightCamera: like LeftCamera but for the player's right eye
# -------------
# Output:
# An extended environment where the agent is shown the pairs of 2d bitmaps
# given by LeftCamera/RightCamera, and is rewarded for acting exactly as if it
# had instead been shown the underlying 3d bitmaps (and is punished otherwise).
def BinocularVision(Game3D, LeftCamera, RightCamera):
  def e(T, play):
    if len(play) == 0:
        # Show the agent the initial game world (as two 2d images)
        empty_action_sequence = []
        actual_3d_world = Game3D(empty_action_sequence)
        left_image = LeftCamera(actual_3d_world)
        right_image = RightCamera(actual_3d_world)
        reward = 0
        obs = cantor_pairing_fnc(left_image, right_image)
        return [reward, obs]

    # The observations in play are pairs of images.
    # Compute a modified play where the observations are 3d matrices.
    modified_play = []
    action_sequence = []
    for roa in [play[i:i+3] for i in range(0,len(play),3)]:
        r, _, a = roa
        modified_obs = Game3D(action_sequence)
        modified_play += [r, modified_obs, a]
        action_sequence += [a]


    # Reward the agent if its most recent action is the same action it
    # would have taken if instead of seeing pairs of 2d images it had
    # seen the equivalent 3d matrix instead.
    prompt, action = play[:-1], play[-1]
    modified_prompt = modified_play[:-1]
    reward = 1 if action == T(modified_prompt) else -1

    # Show the agent the updated game world (as pair of 2d images)
    updated_3d_world = Game3D(action_sequence)
    obs_left = LeftCamera(updated_3d_world)
    obs_right = RightCamera(updated_3d_world)
    obs = cantor_pairing_fnc(obs_left, obs_right)

    return [reward, obs]

  return e