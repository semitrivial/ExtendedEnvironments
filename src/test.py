from test.tests import *
from test.ad_hoc_tests import run_ad_hoc_tests
from test.test_vanilla import test_vanilla

run_ad_hoc_tests()
test_vanilla()

test_incentivize_zero()
test_guarded_treasures()
test_deja_vu()
test_crying_baby()
test_ignore_rewards()
test_false_memories()
test_backward_consciousness()
test_punish_slow_agent()
test_punish_fast_agent()