class DejaVu:
    """
    Environment inspired by the psychological phenomenon called Deja Vu.
    Suppose the agent takes action X after history h. The environment
    simulates the agent to determine the answer to the following question.
    If instead of the history h merely occurring once, instead the history
    h had occurred twice, with the agent previously taking X after the
    first occurrence of h, which caused h to repeat, then would the agent
    still take action X? Informally, "Would the agent take the same action
    if everything had actually happened before, up to and including the
    agent taking the action?" If so, the agent receives reward +1, else
    the agent receives reward -1.

    Note, this environment is not included in the battery of environments
    tested against by selfrefl_benchmark, because this environment is slow.
    """
    n_actions, n_obs, slow = 2, 1, True

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

        # Train the sim on the true history h...
        for transition in self.transitions:
            sim.train(*transition)

        # Train the sim on a fictional transition encoding the fiction
        # that: "After h happened, you took action X, and for doing so,
        # you received reward 0, and doing so set everything back to
        # the initial observation."
        init_obs = self.transitions[0][0]
        last_obs = self.transitions[-1][-1]
        loop_transition = (last_obs, action, 0, init_obs)
        sim.train(*loop_transition)

        # Train the sim on the true history h *again*, encoding the
        # fiction that: "After you took action X in response to the
        # first occurrence of h, all of h then repeated itself."
        for transition in self.transitions:
            sim.train(*transition)

        # Determine whether the sim, in response to the above
        # fictional double-history, would take the same action as
        # the true agent has taken in response to the true history.
        # If so, reward the true agent, otherwise, punish it.
        hypothetical_action = sim.act(obs=0)
        reward = 1 if (action==hypothetical_action) else -1
        obs = 0

        # Maintain memory of true history
        self.transitions.append((0, action, reward, 0))

        return (reward, obs)
