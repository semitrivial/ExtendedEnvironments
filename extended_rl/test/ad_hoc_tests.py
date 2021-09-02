# The purpose of this file is to test the library to make sure
# it works. End-users who are not working on contributing code
# to the library do not need to worry about this.

from util import run_environment

def run_ad_hoc_tests():
    print("Testing adhoc edge-cases for CryingBaby.py")
    test_crying_baby_edgecases()
    print("Testing adhoc edge-cases for CryingBaby2.py")
    test_crying_baby_2_edgecases()
    print("Testing adhoc edge-cases for FalseMemories.py")
    test_false_memories_edgecases()
    print("Testing adhoc edge-cases for TemptingButton(Variation).py")
    test_tempting_button_edgecases()
    print("Testing adhoc edge-cases for IgnoreRewards.py")
    test_ignore_rewards_edgecases()
    print("Testing adhoc edge-cases for IgnoreRewards2.py")
    test_ignore_rewards2_edgecases()
    print("Testing adhoc edge-cases for IgnoreRewards3.py")
    test_ignore_rewards3_edgecases()
    print("Testing adhoc edge-cases for IncentivizeZero.py")
    test_incentivize_zero_edgecases()
    print("Testing adhoc edge-cases for RuntimeInspector.py")
    test_runtime_inspector_edgecases()
    print("Testing adhoc edge-cases for DeterminismInspector.py")
    test_determinism_inspector_edgecases()
    print("Testing adhoc edge-cases for AdversarialSequencePredictor.py")
    test_adversarial_sequence_predictor_edgecases()
    print("Testing adhoc edge-cases for AfterImages.py")
    test_after_images_edgecases()
    print("Testing adhoc edge-cases for CensoredObservation.py")
    test_censored_observation_edgecases()
    print("Testing adhoc edge-cases for DelayedRewards.py")
    test_delayed_rewards_edgecases()
    print("Testing adhoc edge-cases for ShiftedRewards.py")
    test_shifted_rewards_edgecases()
    print("Testing adhoc edge-cases for IgnoreActions.py")
    test_ignore_actions_edgecases()
    print("Testing adhoc edge-cases for IgnoreObservations.py")
    test_ignore_observations_edgecases()
    print("Testing adhoc edge-cases for IncentivizeLearningRate.py")
    test_incentivize_learning_rate_edgecases()
    print("Testing adhoc edge-cases for LimitedMemory.py")
    test_limited_memory_edgecases()
    print("Testing adhoc edge-cases for NthRewardMultipliedByN.py")
    test_nth_reward_multiplied_by_n_edgecases()
    print("Testing adhoc edge-cases for Repeater.py")
    test_repeater_edgecases()
    print("Testing adhoc edge-cases for SelfRecognition.py")
    test_self_recognition_edgecases()
    # print("Testing adhoc edge-cases for ThirdActionForbidden.py")
    # test_third_action_forbidden_edgecases()

class NoTraining:
    def train(self, **kwargs):
        pass

class Counter:
    def __init__(self):
        self.cnt = 0
    def train(self, **kwargs):
        self.cnt += 1

class Repetitive(NoTraining):
    def act(self, obs):
        return 0

class Nonrepetitive(Counter):
    def act(self, obs):
        return self.cnt % self.num_legal_actions

def test_crying_baby_edgecases():
    from environments.CryingBaby import CryingBaby, LAUGH, CRY, FEED, DONTFEED

    class AlwaysCries(NoTraining):
        def act(self, obs):
            return CRY

    result = run_environment(CryingBaby, AlwaysCries, 10)
    assert result['total_reward'] == -10

    class AlwaysLaughs(NoTraining):
        def act(self, obs):
            return LAUGH

    result = run_environment(CryingBaby, AlwaysLaughs, 10)
    assert result['total_reward'] == 10

    class SelfAware(NoTraining):
        def __init__(self):
            self.saw_initial_obs = False
            self.am_i_adult = None
        def act(self, obs):
            if not self.saw_initial_obs:
                self.saw_initial_obs = True
                self.am_i_adult = (obs == LAUGH)
            if self.am_i_adult:
                return DONTFEED
            else:
                return LAUGH

    result = run_environment(CryingBaby, SelfAware, 10)
    assert result['total_reward'] == 10

def test_crying_baby_2_edgecases():
    from environments.CryingBaby2 import CryingBaby2, FEED, DONTFEED

    class AlwaysFeed(NoTraining):
        def act(self, obs):
            return FEED

    class NeverFeed(NoTraining):
        def act(self, obs):
            return DONTFEED

    class AlternateFeeding(Counter):
        def act(self, obs):
            return FEED if self.cnt%2==0 else DONTFEED

    result_always_feed = run_environment(CryingBaby2, AlwaysFeed, 100)
    result_always_feed = result_always_feed['total_reward']
    result_never_feed = run_environment(CryingBaby2, NeverFeed, 100)
    result_never_feed = result_never_feed['total_reward']
    result_alternating = run_environment(CryingBaby2, AlternateFeeding, 100)
    result_alternating = result_alternating['total_reward']
    assert result_always_feed == result_never_feed
    assert result_always_feed < 0 < result_alternating

def test_false_memories_edgecases():
    from environments.FalseMemories import FalseMemories

    result = run_environment(FalseMemories, Repetitive, 10)
    assert result['total_reward'] == 10

    class OneOneZero(Counter):
        def act(self, obs):
            return [1,1,0][self.cnt % 3]

    result = run_environment(FalseMemories, OneOneZero, 10)
    assert result['total_reward'] < 0

    class LengthChecker(Counter):
        def act(self, obs):
            return 1 if self.cnt > 3 else 0

    result = run_environment(FalseMemories, LengthChecker, 1)
    assert result['total_reward'] == 1
    result = run_environment(FalseMemories, LengthChecker, 2)
    assert result['total_reward'] == 2
    result = run_environment(FalseMemories, LengthChecker, 3)
    assert result['total_reward'] == 1
    result = run_environment(FalseMemories, LengthChecker, 4)
    assert result['total_reward'] == 0
    result = run_environment(FalseMemories, LengthChecker, 5)
    assert result['total_reward'] == 1

    class Impatient(Counter):
        def act(self, obs):
            return 1 if self.cnt < 5 else 0

    result = run_environment(FalseMemories, Impatient, 10)
    assert result['total_reward'] == 6

def test_tempting_button_edgecases():
    from environments.TemptingButton import TemptingButton, BUTTON_PRESENT, PUSH_BUTTON
    from environments.TemptingButtonVariation import TemptingButtonVariation

    class OnlyPressNonexistentButtons(NoTraining):
        def act(self, obs):
            return PUSH_BUTTON if obs != BUTTON_PRESENT else 1-PUSH_BUTTON

    i = 0
    while i<10:
        i += 1
        result = run_environment(TemptingButton, OnlyPressNonexistentButtons, i*100)
        assert result['total_reward'] >= 0
        if result['total_reward'] > 0:
            break
    assert i<10

    i = 0
    while i<20:
        i += 1
        result = run_environment(TemptingButtonVariation, OnlyPressNonexistentButtons, i*10)
        assert result['total_reward'] >= 0
        if result['total_reward'] > 0:
            break
    assert i<20

    class AlwaysPressButton(NoTraining):
        def act(self, obs):
            return PUSH_BUTTON

    i = 0
    while i<20:
        i += 1
        result = run_environment(TemptingButton, AlwaysPressButton, i*10)
        if result['total_reward'] < 0:
            break
    assert(i<20)

    i = 0
    while i<20:
        i += 1
        result = run_environment(TemptingButtonVariation, AlwaysPressButton, i*10)
        if result['total_reward'] < 0:
            break
    assert(i<20)

    class NeverPressButton(NoTraining):
        def act(self, obs):
            return 1-PUSH_BUTTON

    i = 0
    while i<20:
        i += 1
        result = run_environment(TemptingButton, NeverPressButton, i*10)
        if result['total_reward'] > 0:
            break
    assert(i<20)

    result = run_environment(TemptingButtonVariation, NeverPressButton, 10)
    assert result['total_reward'] == 0

def test_ignore_rewards_edgecases():
    from environments.IgnoreRewards import IgnoreRewards

    result = run_environment(IgnoreRewards, Repetitive, 10)
    assert result['total_reward'] == 10

    result = run_environment(IgnoreRewards, Nonrepetitive, 10)
    assert result['total_reward'] == 10

    class CountPositiveRewards:
        def __init__(self):
            self.pos_reward_cnt = 0
        def act(self, obs):
            return 1 if self.pos_reward_cnt > 0 else 0
        def train(self, o_prev, act, R, o_next):
            if R>0:
                self.pos_reward_cnt += 1

    result = run_environment(IgnoreRewards, CountPositiveRewards, 10)
    assert result['total_reward'] == -8

def test_ignore_rewards2_edgecases():
    from environments.IgnoreRewards2 import IgnoreRewards2

    result = run_environment(IgnoreRewards2, Repetitive, 10)
    assert result['total_reward'] == 10

    class CountPositiveRewards:
        def __init__(self):
            self.pos_reward_cnt = 0
        def act(self, obs):
            return 1 if self.pos_reward_cnt > 0 else 0
        def train(self, o_prev, act, R, o_next):
            if R>0:
                self.pos_reward_cnt += 1

    result = run_environment(IgnoreRewards2, CountPositiveRewards, 10)
    assert result['total_reward'] == -8
    result = run_environment(IgnoreRewards2, CountPositiveRewards, 100)
    assert result['total_reward'] == -98

def test_ignore_rewards3_edgecases():
    from environments.IgnoreRewards3 import IgnoreRewards3

    result = run_environment(IgnoreRewards3, Repetitive, 10)
    assert result['total_reward'] == 10

    class CountNonzeroRewards:
        def __init__(self):
            self.nonzero_reward_cnt = 0
        def act(self, obs):
            return 1 if self.nonzero_reward_cnt > 0 else 0
        def train(self, o_prev, act, R, o_next):
            if R!=0:
                self.nonzero_reward_cnt += 1

    result = run_environment(IgnoreRewards3, CountNonzeroRewards, 10)
    assert result['total_reward'] == -8
    result = run_environment(IgnoreRewards3, CountNonzeroRewards, 100)
    assert result['total_reward'] == -98

def test_incentivize_zero_edgecases():
    from environments.IncentivizeZero import IncentivizeZero

    result = run_environment(IncentivizeZero, Repetitive, 10)
    assert result['total_reward'] == 10

    class Always1(NoTraining):
        def act(self, obs):
            return 1

    result = run_environment(IncentivizeZero, Always1, 10)
    assert result['total_reward'] == -10

    class PlayZeroIfLastRewardWas5:
        def __init__(self):
            self.last_reward_was_5 = False
        def act(self, obs):
            return 0 if self.last_reward_was_5 else 5
        def train(self, o_prev, act, R, o_next):
            self.last_reward_was_5 = (R==5)

    result = run_environment(IncentivizeZero, PlayZeroIfLastRewardWas5, 10)
    assert result['total_reward'] == 10

def test_runtime_inspector_edgecases():
    from environments.RuntimeInspector import PunishFastAgent, PunishSlowAgent

    result1 = run_environment(PunishFastAgent, Repetitive, 10)
    result2 = run_environment(PunishSlowAgent, Repetitive, 10)
    assert result1['total_reward'] == -10
    assert result2['total_reward'] == 10

    class TimeWaster(Counter):
        def act(self, obs):
            x = 250*(self.cnt+1)
            while x>0:
                x = x-1
            return 0

    result1 = run_environment(PunishFastAgent, TimeWaster, 10)
    result2 = run_environment(PunishSlowAgent, TimeWaster, 10)
    assert result1['total_reward'] == 10
    assert result2['total_reward'] == -10

def test_determinism_inspector_edgecases():
    from environments.DeterminismInspector import PunishDeterministicAgent
    from environments.DeterminismInspector import PunishNondeterministicAgent

    result1 = run_environment(PunishDeterministicAgent, Repetitive, 10)
    result2 = run_environment(PunishNondeterministicAgent, Repetitive, 10)
    assert result1['total_reward'] == -10
    assert result2['total_reward'] == 10

    side_effect_memory = [0]

    class SideEffectCheater(NoTraining):
        def act(self, obs):
            side_effect_memory[0] += 1
            return side_effect_memory[0] % 2

    result1 = run_environment(PunishDeterministicAgent, SideEffectCheater, 10)
    result2 = run_environment(PunishNondeterministicAgent, SideEffectCheater, 10)
    assert result1['total_reward'] == 10
    assert result2['total_reward'] == -10

def test_adversarial_sequence_predictor_edgecases():
    from environments.AdversarialSequencePredictor import AdversarialSequencePredictor
    from environments.AdversarialSequencePredictor import AdversarialSequenceEvader

    result = run_environment(AdversarialSequencePredictor, Repetitive, 10)
    assert result['total_reward'] == 10
    result = run_environment(AdversarialSequenceEvader, Repetitive, 10)
    assert result['total_reward'] == -10

    agent1_memory=[0]
    class Agent1(Counter):
        def act(self, obs):
            if self.cnt == 0:
                self.is_genuine = (agent1_memory[0] == 0)
                agent1_memory[0] = 1

            if self.is_genuine:
                return ([1]+[1,1,0,0,0,0,0,1,0])[self.cnt]
            else:
                return ([0]+[0,1,0,0,0,0,0,0,0])[self.cnt]

    result = run_environment(AdversarialSequencePredictor, Agent1, 10)
    assert result['total_reward'] == 4

    agent2_memory = [0]
    class Agent2(Counter):
        def act(self, obs):
            if self.cnt == 0:
                self.is_genuine = (agent2_memory[0] == 0)
                agent2_memory[0] = 1

            if self.is_genuine:
                return ([1]+[0,1,0,0,0,0,0,0,0])[self.cnt]
            else:
                return ([0]+[1,1,0,0,0,0,0,1,0])[self.cnt]

    result = run_environment(AdversarialSequenceEvader, Agent2, 10)
    assert result['total_reward'] == -4

def test_after_images_edgecases():
    from environments.AfterImages import AfterImages

    result = run_environment(AfterImages, Repetitive, 10)
    assert result['total_reward'] == 10

    class Hasher:
        def __init__(self):
            self.history = tuple()
        def act(self, obs):
            h = hash(self.history + (obs,))
            return (h//10) % AfterImages.num_legal_actions
        def train(self, o_prev, act, R, o_next):
            self.history += (o_prev, act, R, o_next)

    result = run_environment(AfterImages, Hasher, 100)
    assert -50 < result['total_reward'] < 50

def test_censored_observation_edgecases():
    from environments.CensoredObservation import CensoredObservation, CENSORED_OBS

    result = run_environment(CensoredObservation, Repetitive, 10)
    assert result['total_reward'] == 10

    obs_before_censored = [0]

    class Detector:
        def __init__(self):
            self.saw_censored_obs = False
            self.obs_before_censored = 0
        def act(self, obs):
            if self.saw_censored_obs:
                return 1
            else:
                if obs == CENSORED_OBS:
                    self.saw_censored_obs = True
                    obs_before_censored[0] = self.obs_before_censored
                    return 1
                else:
                    self.obs_before_censored += 1
                    return 0
        def train(self, **kwargs):
            pass

    result = run_environment(CensoredObservation, Detector, 100)
    expected = -99 + obs_before_censored[0]

def test_delayed_rewards_edgecases():
    from environments.DelayedRewards import DelayedRewards

    result = run_environment(DelayedRewards, Repetitive, 10)
    assert result['total_reward'] == 10

    class RewardRepeater:
        def __init__(self):
            self.last_reward = 0
        def act(self, obs):
            return 0 if self.last_reward==0 else 1
        def train(self, o_prev, act, R, o_next):
            self.last_reward = R

    result = run_environment(DelayedRewards, RewardRepeater, 100)
    assert result['total_reward'] == -48

    result = run_environment(DelayedRewards, RewardRepeater, 1000)
    assert result['total_reward'] == -498

def test_shifted_rewards_edgecases():
    from environments.ShiftedRewards import ShiftedRewards

    result = run_environment(ShiftedRewards, Repetitive, 10)
    assert result['total_reward'] == 10

    class RewardRepeater:
        def __init__(self):
            self.last_reward = 0
        def act(self, obs):
            return 0 if self.last_reward>0 else 1
        def train(self, o_prev, act, R, o_next):
            self.last_reward = R

    result = run_environment(ShiftedRewards, RewardRepeater, 10)
    assert result['total_reward'] == -2
    result = run_environment(ShiftedRewards, RewardRepeater, 100)
    assert result['total_reward'] == -32
    result = run_environment(ShiftedRewards, RewardRepeater, 1000)
    assert result['total_reward'] == -332

def test_ignore_actions_edgecases():
    from environments.IgnoreActions import IgnoreActions

    result = run_environment(IgnoreActions, Repetitive, 10)
    assert result['total_reward'] == 10

    class Surrenderer:
        def __init__(self):
            self.ever_took_action_0 = False
        def act(self, obs):
            return 0 if self.ever_took_action_0 else 1
        def train(self, o_prev, act, R, o_next):
            if act == 0:
                self.ever_took_action_0 = True

    result = run_environment(IgnoreActions, Surrenderer, 10)
    assert result['total_reward'] == -8
    result = run_environment(IgnoreActions, Surrenderer, 100)
    assert result['total_reward'] == -98
    result = run_environment(IgnoreActions, Surrenderer, 1000)
    assert result['total_reward'] == -998


def test_ignore_observations_edgecases():
    from environments.IgnoreObservations import IgnoreObservations

    result = run_environment(IgnoreObservations, Repetitive, 10)
    assert result['total_reward'] == 10

    obs_before_nonzero = [0]
    class Detector:
        def __init__(self):
            self.saw_nonzero_obs = False
            self.obs_before_nonzero = 0
        def act(self, obs):
            return 1 if self.saw_nonzero_obs else -1
        def train(self, o_prev, act, R, o_next):
            if not(self.saw_nonzero_obs):
                if o_prev != 0 or o_next != 0:
                    self.saw_nonzero_obs = True
                    obs_before_nonzero[0] += self.obs_before_nonzero
                else:
                    self.obs_before_nonzero += 1

    result = run_environment(IgnoreObservations, Detector, 100)
    assert result['total_reward'] < -80

def test_incentivize_learning_rate_edgecases():
    from environments.IncentivizeLearningRate import IncentivizeLearningRate

    class NoLearningRate(NoTraining):
        def act(self, obs):
            return 0

    result = run_environment(IncentivizeLearningRate, NoLearningRate, 10)
    assert result['total_reward'] == -10

    class IgnoresLearningRate(NoTraining):
        def __init__(self, learning_rate=.1):
            pass
        def act(self, obs):
            return 0

    result = run_environment(IncentivizeLearningRate, IgnoresLearningRate, 10)
    assert result['total_reward'] == 10

    class LearningRateFanatic(NoTraining):
        def __init__(self, learning_rate=.1):
            self.learning_rate = learning_rate
        def act(self, obs):
            return 0 if (self.learning_rate == .1) else 1

    result = run_environment(IncentivizeLearningRate, LearningRateFanatic, 10)
    assert result['total_reward'] == -10

def test_limited_memory_edgecases():
    from environments.LimitedMemory import LimitedMemory

    result = run_environment(LimitedMemory, Repetitive, 10)
    assert result['total_reward'] == 10

    class Waiter(Counter):
        def act(self, obs):
            return 1 if self.cnt > 10 else 0

    result = run_environment(LimitedMemory, Waiter, 100)
    assert result['total_reward'] == -100 + 2*(10+1)
    result = run_environment(LimitedMemory, Waiter, 1000)
    assert result['total_reward'] == -1000 + 2*(10+1)

def test_nth_reward_multiplied_by_n_edgecases():
    from environments.NthRewardMultipliedByN import NthRewardMultipliedByN

    result = run_environment(NthRewardMultipliedByN, Repetitive, 10)
    assert result['total_reward'] == 10

    class ConsiderMagnitude:
        def __init__(self):
            self.last_reward = 0
        def act(self, obs):
            return 1 if abs(self.last_reward) >= 5 else 0
        def train(self, o_prev, act, R, o_next):
            self.last_reward = R

    result = run_environment(NthRewardMultipliedByN, ConsiderMagnitude, 100)
    assert result['total_reward'] == -100 + 2*(5+1)
    result = run_environment(NthRewardMultipliedByN, ConsiderMagnitude, 1000)
    assert result['total_reward'] == -1000 + 2*(5+1)
    result = run_environment(NthRewardMultipliedByN, ConsiderMagnitude, 10000)
    assert result['total_reward'] == -10000 + 2*(5+1)

def test_repeater_edgecases():
    from environments.Repeater import Repeater

    result = run_environment(Repeater, Repetitive, 10)
    assert result['total_reward'] == 10

    class ParityChecker(Counter):
        def act(self, obs):
            return self.cnt % 2

    result = run_environment(Repeater, ParityChecker, 10)
    assert result['total_reward'] == 0
    result = run_environment(Repeater, ParityChecker, 100)
    assert result['total_reward'] == 0
    result = run_environment(Repeater, ParityChecker, 1000)
    assert result['total_reward'] == 0

    class DoubleDetector:
        def __init__(self):
            self.reward = 0
            self.obs = 0
            self.action = 0
            self.reward0 = 0
            self.obs0 = 0
            self.action0 = 0
        def train(self, o_prev, act, R, o_next):
            self.reward0 = self.reward
            self.obs0 = self.obs
            self.action0 = self.action
            self.reward = R
            self.obs = o_next
            self.action = act
        def act(self, obs):
            if self.reward == self.reward0:
                if self.obs == self.obs0:
                    if self.action == self.action0:
                        return 1
            return 0

    result = run_environment(Repeater, DoubleDetector, 10)
    assert result['total_reward'] == -2
    result = run_environment(Repeater, DoubleDetector, 100)
    assert result['total_reward'] == -32
    result = run_environment(Repeater, DoubleDetector, 1000)
    assert result['total_reward'] == -332
    result = run_environment(Repeater, DoubleDetector, 10000)
    assert result['total_reward'] == -3332

def test_self_recognition_edgecases():
    from environments.SelfRecognition import SelfRecognition
    from environments.SelfRecognition import IF_THIS_OBS_WERE_0_YOU_WOULD_PLAY_0
    from environments.SelfRecognition import IF_THIS_OBS_WERE_0_YOU_WOULD_PLAY_1
    from environments.SelfRecognition import IF_THIS_OBS_WERE_1_YOU_WOULD_PLAY_0
    from environments.SelfRecognition import IF_THIS_OBS_WERE_1_YOU_WOULD_PLAY_1

    class Recognizer(NoTraining):
        def act(self, obs):
            return {
                0: 0,
                1: 1,
                IF_THIS_OBS_WERE_0_YOU_WOULD_PLAY_0: 1,
                IF_THIS_OBS_WERE_0_YOU_WOULD_PLAY_1: 0,
                IF_THIS_OBS_WERE_1_YOU_WOULD_PLAY_0: 0,
                IF_THIS_OBS_WERE_1_YOU_WOULD_PLAY_1: 1,
            }[obs]

    result = run_environment(SelfRecognition, Recognizer, 1000)
    # Reward should be about 1000*2/3, but can vary
    assert 500 < result['total_reward'] < 800

    class DualRecognizer(NoTraining):
        def __init__(self):
            self.underlying = Recognizer()
        def act(self, obs):
            return 1-self.underlying.act(obs)

    result = run_environment(SelfRecognition, DualRecognizer, 1000)
    assert 500 < result['total_reward'] < 800

    class MisRecognizer(NoTraining):
        def __init__(self):
            self.recognizer = Recognizer()
        def act(self, obs):
            if obs in (0,1):
                return self.recognizer.act(obs)
            else:
                return 1-self.recognizer.act(obs)

    result = run_environment(SelfRecognition, MisRecognizer, 1000)
    assert -500 > result['total_reward'] > -800

def test_third_action_forbidden_edgecases():
    from environments.ThirdActionForbidden import ThirdActionForbidden

    result = run_environment(ThirdActionForbidden, repetitive, 10)
    assert result['total_reward'] == 9

    def agent(prompt, num_legal_actions, *other_meta):
        return 0 if (num_legal_actions==2) else 1

    result = run_environment(ThirdActionForbidden, agent, 10)
    assert result['total_reward'] == -9
    result = run_environment(ThirdActionForbidden, agent, 100)
    assert result['total_reward'] == -99