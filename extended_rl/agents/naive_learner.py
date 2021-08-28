from prerandom import agentrandom


class NaiveLearner1:
    def __init__(self):
        self.n_actions = self.num_legal_actions
        self.rnd_counter = 0
        self.best_action = 0
        self.best_reward = 0

    def act(self, obs):
        if agentrandom.random(self.rnd_counter) < .15:
            return agentrandom.randrange(self.n_actions, self.rnd_counter+1)

        return self.best_action

    def train(self, o_prev, act, R, o_next):
        self.rnd_counter += 2
        if R > self.best_reward:
            self.best_reward = R
            self.best_action = act

class NaiveLearner2:
    def __init__(self):
        self.n_actions = self.num_legal_actions
        self.n_obs = self.num_possible_obs
        self.rnd_counter = 0
        self.bests = {o: {'a': 0, 'r': 0} for o in range(self.n_obs)}

    def act(self, obs):
        if agentrandom.random(self.rnd_counter) < .15:
            return agentrandom.randrange(self.n_actions, self.rnd_counter+1)

        return self.bests[obs]['a']

    def train(self, o_prev, act, R, o_next):
        self.rnd_counter += 2
        if R > self.bests[o_prev]['r']:
            self.bests[o_prev] = {'a': act, 'r': R}

class NaiveLearner3:
    def __init__(self):
        self.n_actions = self.num_legal_actions
        self.actions = list(range(self.n_actions))
        self.rnd_counter = 0
        self.total_rewards = {a: 0 for a in self.actions}
        self.act_counts = {a: 0 for a in self.actions}
        self.avg_rewards = {a: 0 for a in self.actions}
        self.best_action = 0

    def act(self, obs):
        if agentrandom.random(self.rnd_counter) < .15:
            return agentrandom.randrange(self.n_actions, self.rnd_counter)

        return self.best_action

    def train(self, o_prev, act, R, o_next):
        self.rnd_counter += 2
        self.total_rewards[act] += R
        self.act_counts[act] += 1
        avg_reward = self.total_rewards[act] / self.act_counts[act]
        self.avg_rewards[act] = avg_reward

        self.best_action = max(
            self.actions, 
            key = lambda i: self.avg_rewards[i]
        )

class NaiveLearner4:
    def __init__(self):
        self.n_actions = self.num_legal_actions
        self.n_obs = self.num_possible_obs
        self.actions = list(range(self.n_actions))
        self.observs = list(range(self.n_obs))
        self.rnd_counter = 0
        self.total_rewards = {
            o: {a: 0 for a in self.actions} for o in self.observs
        }
        self.act_counts = {
            o: {a: 0 for a in self.actions} for o in self.observs
        }
        self.avg_rewards = {
            o: {a: 0 for a in self.actions} for o in self.observs
        }
        self.best_actions = {o: 0 for o in self.observs}

    def act(self, obs):
        if agentrandom.random(self.rnd_counter) < .15:
            return agentrandom.randrange(self.n_actions, self.rnd_counter+1)

        return self.best_actions[obs]

    def train(self, o_prev, act, R, o_next):
        self.rnd_counter += 2
        self.total_rewards[o_prev][act] += R
        self.act_counts[o_prev][act] += 1
        total_reward = self.total_rewards[o_prev][act]
        avg_reward = total_reward / self.act_counts[o_prev][act]
        self.avg_rewards[o_prev][act] = avg_reward

        self.best_actions[o_prev] = max(
            self.actions, 
            key = lambda i: self.avg_rewards[o_prev][i]
        )



