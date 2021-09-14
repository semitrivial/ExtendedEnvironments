from extended_rl.prerandom import agentrandom


class Q_learner:
  """
  Basic Q-learning agent, see https://en.wikipedia.org/wiki/Q-learning.
  """
  def __init__(self, epsilon=0.9, learning_rate=0.1, gamma=0.9, **kwags):
    self.epsilon = epsilon
    self.learning_rate = learning_rate
    self.gamma = gamma
    self.n_actions = self.num_legal_actions
    self.actions = range(self.n_actions)
    self.qtable = {}
    self.rand_counter = 0

  def act(self, obs):
    qtable, epsilon, actions = self.qtable, self.epsilon, self.actions
    maybe_add_obs_to_qtable(qtable, actions, obs)

    if agentrandom.random(self.rand_counter) > epsilon:
      return agentrandom.randrange(self.n_actions, self.rand_counter+1)
    elif all(qtable[obs,a]==0 for a in actions):
      return agentrandom.randrange(self.n_actions, self.rand_counter+1)
    else:
      return max(actions, key=lambda a: qtable[obs,a])

  def train(self, o_prev, a, r, o_next):
    qtable, actions, gamma = self.qtable, self.actions, self.gamma
    maybe_add_obs_to_qtable(qtable, actions, o_prev)
    maybe_add_obs_to_qtable(qtable, actions, o_next)
    qtarget = r + gamma * max([qtable[o_next,actn] for actn in actions])
    qpredict = qtable[o_prev, a]
    qtable[o_prev, a] += self.learning_rate * (qtarget - qpredict)
    self.rand_counter += 2

def maybe_add_obs_to_qtable(qtable, actions, obs):
  if not((obs, 0) in qtable):
    qtable.update({(obs, a): 0 for a in actions})