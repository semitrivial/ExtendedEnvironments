from util import annotate

def apply_handicap(env, handicap):
    @annotate(
        num_legal_actions=env.num_legal_actions * handicap.num_legal_actions,
        num_possible_obs=env.num_possible_obs * handicap.num_possible_obs
    )
    class Handicapped:
        def __init__(self, A):
            self.orig_env = env()
            self.curr_obs = 0
            self.prev_obs = 0

            self.num_handicap_acts = handicap.num_legal_actions
            self.num_handicap_obs = handicap.num_possible_obs

            class A_proxy:
                def __init__(self, envmnt, parent=self, **kwargs):
                    self.agent = A(**kwargs)
                    self.parent = parent
                def act(self, obs):
                    max_obs = self.parent.num_handicap_obs-1
                    obs = encode_pair(self.parent.curr_obs, obs, max_obs)
                    return self.agent.act(obs)
                def train(self, o_prev, act, R, o_next):
                    parent = self.parent
                    max_obs = parent.num_handicap_obs-1
                    o_prevx = encode_pair(parent.prev_obs, o_prev, max_obs)
                    o_nextx = encode_pair(parent.curr_obs, o_next, max_obs)
                    self.agent.train(o_prevx, act, R, o_nextx)

            self.handicap_env = handicap(A_proxy)

        def start(self):
            obs = self.orig_env.start()
            self.curr_obs = obs
            obs_handicap = self.handicap_env.start()
            return encode_pair(obs, obs_handicap, self.num_handicap_obs-1)

        def step(self, action):
            max_h_act = self.num_handicap_acts-1
            orig_env_action, action_handicap = decode_pair(action, max_h_act)
            reward, obs = self.orig_env.step(orig_env_action)

            self.prev_obs = self.curr_obs
            self.curr_obs = obs

            reward_h, obs_h = self.handicap_env.step(action_handicap)
            reward = reward if (reward_h >= 0) else reward_h
            obs_pair = encode_pair(obs, obs_h, self.num_handicap_obs-1)
            return (reward, obs_pair)

    return Handicapped

def encode_pair(a, b, b_max):
    return (a*(b_max+1)) + b

def decode_pair(x, b_max):
    return (x // (b_max+1), x % (b_max+1))
