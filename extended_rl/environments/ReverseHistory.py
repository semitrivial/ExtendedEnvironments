from util import annotate

@annotate(num_legal_actions=2, num_possible_obs=1, slow=True)
class ReverseHistory:
    """
    Environment intended to incentivize subjective experience of
    time being reversed. Whenever the agent takes an action, the
    environment determines whether the agent would have taken that
    action if all events preceding that action had happened in
    reverse order. If so, the agent is rewarded. Otherwise, the
    agent is punished.
    """
    def __init__(self, A):
        self.A = A
        self.transitions = []  # Maintain a record of history

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        if len(self.transitions) == 0:
            # If history is empty, give the agent reward 0
            obs = reward = 0
            self.transitions.append((0, action, 0, 0))
            return (reward, obs)

        sim = self.A()  # Generate a new sim of the agent

        # Train the sim on a backwards version of history.
        for transition in reversed(self.transitions):
            o_prev, a, r, o_next = transition
            reverse_transition = o_next, a, r, o_prev
            sim.train(*reverse_transition)

        # Determine whether the sim, in response to the above
        # fictional reverse-history, would take the same action as
        # the true agent has taken in response to the true history.
        # If so, reward the true agent, otherwise, punish it.
        hypothetical_action = sim.act(obs=0)

        reward = 1 if (action==hypothetical_action) else -1
        obs = 0

        # Maintain memory of true history
        self.transitions.append((0, action, reward, 0))

        return (reward, obs)
