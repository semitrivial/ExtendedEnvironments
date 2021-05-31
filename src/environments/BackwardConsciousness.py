class BackwardConsciousness:
    """
    Environment intended to incentivize subjective experience of
    time being reversed. Whenever the agent takes an action, the
    environment determines whether the agent would have taken that
    action if all events preceding that action had happened in
    reverse order. If so, the agent is rewarded. Otherwise, the
    agent is punished.
    """
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1

    def react(self, T, play):
        if len(play) == 0:
            reward, obs = 0, 0
            return (reward, obs)

        action = play[-1]
        prompt = reverse_prompt(play)

        reward = 1 if action == T(prompt) else -1
        obs = 0
        return (reward, obs)

def reverse_prompt(play):
    """
    Given a sequence of the form r_0,o_0,a_0,...,r_n,o_n,a_n,
    discard the final a_n and reverse the remaining sequence,
    while preserving the reward-observation-action order, so
    as to obtain:
    r_n,o_n,a_{n-1},r_{n-1},o_{n-1},a_{n-2},...,r_1,o_1,a_0,r_0,o_0
    """

    # First, break the play into a list of Reward-Observation-Action tiples
    triples = [play[i:i+3] for i in range(0,len(play),3)]

    # Reverse the list of triples
    triples.reverse()

    # Turn each Reward-Obs-Action triple into Action-Reward-Obs
    triples = [(a,r,o) for (r,o,a) in triples]

    # Remove the final action (now the 0th action due to reversing)
    triples[0] = (triples[0][1], triples[0][2])

    # Combine the list-of-lists and return the result
    return tuple(j for i in triples for j in i)
