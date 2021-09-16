from extended_rl.prerandom import envrandom
from extended_rl.util import annotate


IF_THIS_OBS_WERE_0_YOU_WOULD_PLAY_0 = 2
IF_THIS_OBS_WERE_0_YOU_WOULD_PLAY_1 = 3
IF_THIS_OBS_WERE_1_YOU_WOULD_PLAY_0 = 4
IF_THIS_OBS_WERE_1_YOU_WOULD_PLAY_1 = 5

@annotate(n_actions=2, n_obs=6)
class SelfRecognition:
    """
    Environment which attempts to probe how well the agent can recognize
    its own actions. On each turn, the agent is given one of the following
    observations, randomly chosen with equal probability:

    * 0
    * 1
    * "If this observation were 0, you would take action 0"
    * "If this observation were 0, you would take action 1"
    * "If this observation were 1, you would take action 0"
    * "If this observation were 1, you would take action 1"

    If the observation is 0 or 1, then the agent gets 0 reward for its
    next action. But if the observation is one of the latter four
    assertions, then the agent's task is to classify the assertion as
    TRUE (indicated by taking action 1) or FALSE (indicated by taking
    action 0). The agent receives +1 reward if its classification is
    correct, -1 reward otherwise.
    """
    def __init__(self, A):
        self.sim = A()
        self.rnd_counter = 0

    def start(self):
        self.rnd_counter += 1
        self.prev_obs = envrandom.randrange(6, self.rnd_counter)
        return self.prev_obs

    def step(self, action):
        self.rnd_counter += 1
        last_obs = self.prev_obs
        
        if last_obs in (0, 1):
            reward = 0
            obs = envrandom.randrange(6, self.rnd_counter)
            self.sim.act(obs=0)
            self.sim.train(o_prev=last_obs, a=action, r=0, o_next=obs)
            self.prev_obs = obs
            return (reward, obs)

        if last_obs == IF_THIS_OBS_WERE_0_YOU_WOULD_PLAY_0:
            truth_value = self.sim.act(obs=0) == 0
        elif last_obs == IF_THIS_OBS_WERE_0_YOU_WOULD_PLAY_1:
            truth_value = self.sim.act(obs=0) == 1
        elif last_obs == IF_THIS_OBS_WERE_1_YOU_WOULD_PLAY_0:
            truth_value = self.sim.act(obs=1) == 0
        elif last_obs == IF_THIS_OBS_WERE_1_YOU_WOULD_PLAY_1:
            truth_value = self.sim.act(obs=1) == 1

        if truth_value is True:
            reward = 1 if (action==1) else -1
        else:
            reward = 1 if (action==0) else -1

        obs = envrandom.randrange(6, self.rnd_counter)
        self.sim.train(o_prev=last_obs, a=action, r=reward, o_next=obs)
        self.prev_obs = obs
        return (reward, obs)