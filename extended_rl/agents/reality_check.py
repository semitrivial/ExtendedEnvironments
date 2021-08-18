def reality_check(A0):
  class A0_RC:
    def __init__(self, env, **kwargs):
      self.underlying = A0(env, **kwargs)
      self.first_action = None
      self.expected_training_action = None
      self.found_unexpected_action = False

    def act(self, obs):
      if self.found_unexpected_action:
        return self.first_action

      if self.first_action is None:
        self.first_action = self.underlying.act(obs)
        self.expected_training_action = self.first_action
        return self.first_action

      self.expected_training_action = self.underlying.act(obs)
      return self.expected_training_action

    def train(self, o_prev, act, R, o_next):
      if not self.found_unexpected_action:
        if act == self.expected_training_action:
          self.underlying.train(o_prev, act, R, o_next)
        else:
          self.found_unexpected_action = True

  return A0_RC