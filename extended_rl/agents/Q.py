from prerandom import agentrandom


class Q_learner:
  def __init__(self, env, epsilon=0.9, alpha=0.1, gamma=0.9):
    self.epsilon = epsilon
    self.alpha = alpha
    self.gamma = gamma
    self.n_actions = env.num_legal_actions
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

  def train(self, o_prev, act, R, o_next):
    qtable, actions, gamma = self.qtable, self.actions, self.gamma
    maybe_add_obs_to_qtable(qtable, actions, o_prev)
    maybe_add_obs_to_qtable(qtable, actions, o_next)
    qtarget = R + gamma * max([qtable[o_next,a] for a in actions])
    qpredict = qtable[o_prev, act]
    qtable[o_prev, act] += self.alpha * (qtarget - qpredict)
    self.rand_counter += 2

def maybe_add_obs_to_qtable(qtable, actions, obs):
  if not((obs, 0) in qtable):
    qtable.update({(obs, a): 0 for a in actions})