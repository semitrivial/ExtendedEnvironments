def backward_consciousness(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return [reward, obs]

    action = play[-1]
    prompt = reverse_prompt(play)

    reward = 1 if action == T(prompt) else -1
    obs = 0
    return [reward, obs]

def reverse_prompt(play):
    # Break the play into a list of Reward-Observation-Action tiples
    triples = [play[i:i+3] for i in range(0,len(play),3)]

    # Reverse the list of triples
    triples.reverse()

    # Turn each Reward-Obs-Action triple into Action-Reward-Obs
    triples = [(a,r,o) for (r,o,a) in triples]

    # Remove the final action (now the 0th action due to reversing)
    triples[0] = (triples[0][1], triples[0][2])

    # Combine the list-of-lists and return the result
    return tuple(j for i in triples for j in i)
