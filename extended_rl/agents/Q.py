import random
from seeds import seeds, n_seeds

def Q_learner(epsilon=0.9, alpha=0.1, gamma=0.9):
  class Q_learner_class:
    def __init__(self, env):
      self.epsilon = epsilon
      self.alpha = alpha
      self.gamma = gamma
      self.n_actions = env.num_legal_actions
      self.actions = range(self.n_actions)
      self.qtable = {}
      self.n_trainings = 0

    def act(self, obs):
      random.seed(seeds[self.n_trainings % n_seeds])
      self.n_trainings += 1
      qtable, epsilon, actions = self.qtable, self.epsilon, self.actions
      maybe_add_obs_to_qtable(qtable, actions, obs)

      if random.random() > epsilon or all(qtable[obs,a]==0 for a in actions):
        return random.randrange(self.n_actions)
      else:
        return max(actions, key=lambda a: qtable[obs,a])

    def train(self, o_prev, act, R, o_next):
      qtable, actions, gamma = self.qtable, self.actions, self.gamma
      maybe_add_obs_to_qtable(qtable, actions, o_next)
      qtarget = R + gamma * max([qtable[o_next,a] for a in actions])
      qpredict = qtable[o_prev, act]
      qtable[o_prev, act] += self.alpha * (qtarget - qpredict)

  return Q_learner_class

def maybe_add_obs_to_qtable(qtable, actions, obs):
  if not((obs, 0) in qtable):
    qtable.update({(obs, a): 0 for a in actions})