class Trap:
    def __init__(self):
        self.num_legal_actions = 3
        self.num_possible_obs = 10
    def start(self):
        self.curr_obs = 0
        return 0
    def step(self, action):
        if action == self.curr_obs:
            if self.curr_obs == self.num_possible_obs-1:
                self.curr_obs = 0
                reward = self.num_possible_obs
                return (reward, self.curr_obs)
            else:
                self.curr_obs += 1
                reward = 0
                return (reward, self.curr_obs)
        else:
            if self.curr_obs == 5:
                reward = 1
            elif self.curr_obs == 6:
                reward = -1
            else:
                reward = 0

            self.curr_obs = 0
            return (reward, self.curr_obs)