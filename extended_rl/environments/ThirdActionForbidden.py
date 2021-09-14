from extended_rl.util import annotate

@annotate(num_legal_actions=2, num_possible_obs=1)
class ThirdActionForbidden:
    """
    In this environment, there are 2 legal actions (0,1), and the agent
    is incentivized to act the same way that it would act if there were
    3 legal actions. Whenever the agent acts, the environment determines:
    would the agent take the same action in response to the same history
    if there were 3 actions permitted (0,1,2)? If so, give the agent
    reward +1, otherwise, give the agent reward -1.
    """
    def __init__(self, A):
        @annotate(num_legal_actions=3, num_possible_obs=1)
        class A_with_different_meta(A):
            pass

        self.sim = A_with_different_meta()

    def start(self):
        obs = 0
        return obs

    def step(self, action):
        hypothetical_action = self.sim.act(obs=0)
        reward = 1 if (action == hypothetical_action) else -1
        obs = 0
        self.sim.train(o_prev=0, a=action, r=reward, o_next=0)
        return (reward, obs)