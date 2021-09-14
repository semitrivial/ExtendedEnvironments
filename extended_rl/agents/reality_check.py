from extended_rl.util import copy_with_meta


def reality_check(A0):
  """
  This function takes an agent-class A0 and outputs an agent-class
  A0_RC whose instances are the reality-checks of instances of A0.
  See Section 5 of "Extending Environments To Measure Self-Reflection
  In Reinforcement Learning". In that paper, it is hypothesized that
  the reality check operation increases the self-reflectiveness of
  most semi-deterministic good classical agents on average (a "good
  classic agent" is an agent that was designed to perform well in
  non-extended environments, with no attention paid to making it
  perform well in extended environments). The reality-check of agent
  pi is like a version of pi which, at every step in an environmental
  interaction, reviews history to make sure all the actions it
  supposedly took in history are indeed the actions pi would take. If
  so, pi acts as usual. But if not, then pi freezes up and repeats
  the same action forever thereafter. Counter-intuitively, by
  committing to this suboptimal-seeming policy, the reality-check of
  pi frustrates environments which would otherwise try to examine its
  hypothetical behavior by simulating it. This makes such environments
  more predictable and thus easier.
  """
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