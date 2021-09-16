from util import run_environment, copy_with_meta

run_environment_backup = run_environment

def run_environment(env, A, num_steps):
    class env_monkey(env):
        def __init__(self, *args, **kwargs):
            env_with_meta = copy_with_meta(env, self)
            self.underlying = env_with_meta(*args, **kwargs)
        def start(self):
            obs = self.underlying.start()
            assert 0 <= obs < self.n_obs
            return obs
        def step(self, action):
            assert 0 <= action < self.n_actions
            reward, obs = self.underlying.step(action)
            assert 0 <= obs < self.n_obs
            return (reward, obs)

    class A_monkey(A):
        def __init__(self, *args, **kwargs):
            A_with_meta = copy_with_meta(A, self)
            self.underlying = A_with_meta(*args, **kwargs)
        def act(self, obs):
            assert 0 <= obs < self.n_obs
            action = self.underlying.act(obs)
            assert 0 <= action < self.n_actions
            return action
        def train(self, o_prev, a, r, o_next):
            assert 0 <= o_prev < self.n_obs
            assert 0 <= o_next < self.n_obs
            assert 0 <= a < self.n_actions
            self.underlying.train(o_prev=o_prev, a=a, r=r, o_next=o_next)

    return run_environment_backup(env_monkey, A_monkey, num_steps)