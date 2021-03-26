def get_SB3_agents():
    import stable_baselines3 as SB3
    import gym
    from gym import Env, spaces

    class prompt_env(Env):
        def __init__(self, prompt):
            super(prompt_env, self).__init__()
            self.action_space = spaces.Discrete(10)
            self.observation_space = spaces.Discrete(10)
            self.prompt = prompt
            self.stepcount = 1
        def reset(self):
            return self.prompt[1]
        def step(self, action):
            i = self.stepcount
            reward, obs = self.prompt[3*i+0], self.prompt[3*i+1]
            done, info = False, {}
            self.stepcount += 1
            return obs, reward, done, info
        def render(self, mode='human'):
            return
        def close(self):
            return

    def agent_A2C(prompt):
        runner = prompt_env(prompt)

        if len(prompt) == 2:
            A = SB3.A2C('MlpPolicy', runner, n_steps=1)
            obs = prompt[1]
            action, _ = A.predict(obs)
            return action

        n_steps = (len(prompt)//3)-1
        A = SB3.A2C('MlpPolicy', runner, n_steps=n_steps)
        A = A.learn(n_steps)
        obs = prompt[-1]
        action, _ = A.predict(obs)
        return action

    return {
        'A2C': agent_A2C
    }
