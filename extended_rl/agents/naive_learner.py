from extended_rl.prerandom import agentrandom

class NaiveLearner:
    def __init__(self, **kwargs):
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

    def train(self, o_prev, a, r, o_next):
        self.rnd_counter += 2
        self.total_rewards[o_prev][a] += r
        self.act_counts[o_prev][a] += 1
        total_reward = self.total_rewards[o_prev][a]
        avg_reward = total_reward / self.act_counts[o_prev][a]
        self.avg_rewards[o_prev][a] = avg_reward

        self.best_actions[o_prev] = max(
            self.actions, 
            key = lambda i: self.avg_rewards[o_prev][i]
        )
