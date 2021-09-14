from util import copy_with_meta


def reality_check(A0):
  class A0_RC:
    def __init__(self, **kwargs):
      A0_with_meta = copy_with_meta(A0, meta_src=self)
      self.underlying = A0_with_meta(**kwargs)
      self.found_unexpected_action = False
      self.first_action = None
      self.act_dict = {}

    def act(self, obs):
      if self.found_unexpected_action:
        return self.first_action

      action = self.underlying.act(obs)
      self.act_dict[obs] = action
      self.first_action = self.first_action or action
      return action

    def train(self, o_prev, a, r, o_next):
      if self.found_unexpected_action:
        return
      if not(o_prev in self.act_dict):
        self.act(o_prev)
      if a == self.act_dict[o_prev]:
        self.underlying.train(o_prev, a, r, o_next)
        self.act_dict.clear()
      else:
        self.found_unexpected_action = True

  A0_RC.__name__ = f'reality_check({A0.__name__})'
  A0_RC.__qualname__ = f'reality_check({A0.__qualname__})'
  return A0_RC