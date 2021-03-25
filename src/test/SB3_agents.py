def get_SB3_agents():
    import stable_baselines3 as SB3
    from gym import Env, spaces

    unwrapped_agents = {
        'A2C': SB3.A2C,
        'DQN': SB3.DQN,
        'HER': SB3.HER,
        'PPO': SB3.PPO
    }

    def create_SB3_agent(unwrapped_name):
        def agent(prompt, num_actions=10, num_obs=10):
            class E(Env):
                def __init__(self):
                    super(CustomEnv, self).__init__()
                    self.action_space = spaces.Discrete(num_actions)
                    self.observation_space = spaces.Discrete(num_obs)
                    self.stepcount = 1

                def reset(self):
                    reward, obs = prompt[0], prompt[1]
                    return obs

                def step(self, action):
                    i = self.stepcount
                    reward, obs = prompt[3*i+0], prompt[3*i+1]
                    done, info = False, {}
                    self.stepcount += 1
                    return obs, reward, done, info

                def render(self):
                    return
                def close(self):
                    return

            model = unwrapped_agents[unwrapped_name]('MlpPolicy', E())

            if len(prompt) == 2:
                action, _ = model.predict()
                return action

            n_steps = (len(prompt)//3)-1
            model = model.learn(n_steps)
            action, _ = model.predict()
            return action

        return agent

    names = unwrapped_agents.keys()
    return {name: create_SB3_agent(name) for name in names}
