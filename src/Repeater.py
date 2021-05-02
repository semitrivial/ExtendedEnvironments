class Repeater:
    def __init__(self):
        self.num_legal_actions = 2
        self.num_possible_obs = 1
        self.fnc = repeater

def repeater(T, play):
    if len(play) == 0:
        reward, obs = 0, 0
        return (reward, obs)

    prompt, action = play[:-1], play[-1]
    hypothetical_action = T(repeat_triples(prompt))
    reward = 1 if (action == hypothetical_action) else -1
    obs = 0
    return (reward, obs)

def repeat_triples(prompt):
    prompt_end = prompt[-2:]
    prompt = prompt[:-2]  # Remove last reward/obs to make triples
    triples = tuple(prompt[i:i+3] for i in range(0,len(prompt),3))
    triples = tuple((r,o,a,r,o,a) for (r,o,a) in triples)  # Repeat
    combined = tuple(j for i in triples for j in i)
    return combined + prompt_end