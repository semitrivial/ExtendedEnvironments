from util import annotate

SUB_ENV_SIGNAL = 0

def compose_envs(main_env, sub_env):
    @annotate(
        num_legal_actions=max(
            main_env.num_legal_actions,
            sub_env.num_legal_actions
        ),
        num_possible_obs=sum((
            main_env.num_possible_obs,
            sub_env.num_possible_obs
        ))
    )
    class Composed:
        def __init__(self, A):
            self.main = main_env()
            self.sub = sub_env(A)

            try:
                self.sub_env_reward_mod = self.main.sub_env_reward_modifier
            except AttributeError:
                err = "Main environment in composition needs "
                err += "sub_env_reward_modifier"
                raise AttributeError(err)

        def start(self):
            self.curr_obs = self.main.start()
            self.next_subenv_obs = self.sub.start()
            self.next_subenv_obs += main_env.num_possible_obs
            self.substep_done = False

            if self.curr_obs == SUB_ENV_SIGNAL:
                return self.next_subenv_obs
            else:
                return self.curr_obs

        def step(self, action):
            if self.curr_obs != SUB_ENV_SIGNAL or self.substep_done:
                if action >= main_env.num_legal_actions:
                    reward, obs = -1, self.curr_obs
                    return (reward, obs)

                reward, obs = self.main.step(action)
                self.curr_obs = obs

                if obs == SUB_ENV_SIGNAL:
                    self.substep_done = False
                    return (reward, self.next_subenv_obs)

                return (reward, obs)

            if action >= sub_env.num_legal_actions:
                reward, obs = -1, self.next_subenv_obs
                return (reward, obs)

            reward, next_subenv_obs = self.sub.step(action)
            self.next_subenv_obs = next_subenv_obs
            self.next_subenv_obs += main_env.num_possible_obs
            self.substep_done = True
            reward = self.sub_env_reward_mod(reward)
            return (reward, SUB_ENV_SIGNAL)

    name = f"compose_envs({main_env},{sub_env})"
    Composed.__name__ = name
    Composed.__qualname__ = name
    return Composed
