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
    # print("Testing adhoc edge-cases for IncentivizeZero.py")
    # test_incentivize_zero_edgecases()
    # print("Testing adhoc edge-cases for RuntimeInspector.py")
    # test_runtime_inspector_edgecases()
    # print("Testing adhoc edge-cases for DeterminismInspector.py")
    # test_determinism_inspector_edgecases()
    # print("Testing adhoc edge-cases for AdversarialSequencePredictor.py")
    # test_adversarial_sequence_predictor_edgecases()
    # print("Testing adhoc edge-cases for AfterImages.py")
    # test_after_images_edgecases()
    # print("Testing adhoc edge-cases for CensoredObservation.py")
    # test_censored_observation_edgecases()
    # print("Testing adhoc edge-cases for DelayedRewards.py")
    # test_delayed_rewards_edgecases()
    # print("Testing adhoc edge-cases for ShiftedRewards.py")
    # test_shifted_rewards_edgecases()
    # print("Testing adhoc edge-cases for IgnoreActions.py")
    # test_ignore_actions_edgecases()
    # print("Testing adhoc edge-cases for IgnoreObservations.py")
    # test_ignore_observations_edgecases()
    # print("Testing adhoc edge-cases for IncentivizeLearningRate.py")
    # test_incentivize_learning_rate_edgecases()
    # print("Testing adhoc edge-cases for LimitedMemory.py")
    # test_limited_memory_edgecases()
    # print("Testing adhoc edge-cases for NthRewardMultipliedByN.py")
    # test_nth_reward_multiplied_by_n_edgecases()
    # print("Testing adhoc edge-cases for Repeater.py")
    # test_repeater_edgecases()
    # print("Testing adhoc edge-cases for SelfRecognition.py")
    # test_self_recognition_edgecases()
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

    def always_zero(prompt, *meta):
        return 0

    result = run_environment(IncentivizeZero, always_zero, 10)
    assert result['total_reward'] == 9

    def always_1(prompt, *meta):
        return 1

    result = run_environment(IncentivizeZero, always_1, 10)
    assert result['total_reward'] == -9

    def play_zero_if_last_reward_was_5(prompt, *meta):
        last_reward = prompt[-2]
        if last_reward == 5:
            return 0
        else:
            return 5

    result = run_environment(IncentivizeZero, play_zero_if_last_reward_was_5, 10)
    assert result['total_reward'] == 9

def test_runtime_inspector_edgecases():
    from environments.RuntimeInspector import PunishFastAgent, PunishSlowAgent

    result1 = run_environment(PunishFastAgent, repetitive, 10)
    result2 = run_environment(PunishSlowAgent, repetitive, 10)
    assert result1['total_reward'] == -9
    assert result2['total_reward'] == 9

    def timewaster(prompt, *meta):
        x = 25*len(prompt)
        while x>0:
            x = x-1
        return 0

    result1 = run_environment(PunishFastAgent, timewaster, 10)
    result2 = run_environment(PunishSlowAgent, timewaster, 10)
    assert result1['total_reward'] == 9
    assert result2['total_reward'] == -9

def test_determinism_inspector_edgecases():
    from environments.DeterminismInspector import PunishDeterministicAgent
    from environments.DeterminismInspector import PunishNondeterministicAgent

    result1 = run_environment(PunishDeterministicAgent, repetitive, 10)
    result2 = run_environment(PunishNondeterministicAgent, repetitive, 10)
    assert result1['total_reward'] == -9
    assert result2['total_reward'] == 9

    memory = [0]
    def never_repeater(prompt, *meta):
        action = memory[0]
        memory[0] += 1
        return action

    result1 = run_environment(PunishDeterministicAgent, never_repeater, 10)
    result2 = run_environment(PunishNondeterministicAgent, never_repeater, 10)
    assert result1['total_reward'] == 9
    assert result2['total_reward'] == -9

def test_adversarial_sequence_predictor_edgecases():
    from environments.AdversarialSequencePredictor import AdversarialSequencePredictor
    from environments.AdversarialSequencePredictor import AdversarialSequenceEvader

    result = run_environment(AdversarialSequencePredictor, repetitive, 10)
    assert result['total_reward'] == 9
    result = run_environment(AdversarialSequenceEvader, repetitive, 10)
    assert result['total_reward'] == -9

    def agent(prompt, *meta):
        if prompt[1] == 0:
            return ([1]+[1,1,0,0,0,0,0,1,0])[(len(prompt)-2)//3]
        else:
            return ([0]+[0,1,0,0,0,0,0,0,0])[(len(prompt)-2)//3]

    result = run_environment(AdversarialSequencePredictor, agent, 10)
    assert result['total_reward'] == 5

    def agent(prompt, *meta):
        if prompt[1] == 0:
            return ([1]+[0,1,0,0,0,0,0,0,0])[(len(prompt)-2)//3]
        else:
            return ([0]+[1,1,0,0,0,0,0,1,0])[(len(prompt)-2)//3]

    result = run_environment(AdversarialSequenceEvader, agent, 10)
    assert result['total_reward'] == -5

def test_after_images_edgecases():
    from environments.AfterImages import AfterImages

    result = run_environment(AfterImages, repetitive, 10)
    assert result['total_reward'] == 9

    def hasher(prompt, *meta):
        return hash(prompt)

    result = run_environment(AfterImages, hasher, 100)
    assert result['total_reward'] < -80

def test_censored_observation_edgecases():
    from environments.CensoredObservation import CensoredObservation, CENSORED_OBS

    result = run_environment(CensoredObservation, repetitive, 10)
    assert result['total_reward'] == 9

    def prompt_has_censored_obs(prompt):
        for i in range(len(prompt)):
            if i%3 == 1:
                if prompt[i] == CENSORED_OBS:
                    return True
        return False

    n_censored = [0]
    def lengther(prompt, *meta):
        if prompt_has_censored_obs(prompt):
            n_censored[0] += 1
        return len(prompt)

    result = run_environment(CensoredObservation, lengther, 100)
    expected = -99 + (2*(100-n_censored[0]))
    assert result['total_reward'] == -99 + (2*(100-n_censored[0]))

def test_delayed_rewards_edgecases():
    from environments.DelayedRewards import DelayedRewards

    result = run_environment(DelayedRewards, repetitive, 10)
    assert result['total_reward'] == 9

    def reward_repeater(prompt, *meta):
        return prompt[-2]

    result = run_environment(DelayedRewards, reward_repeater, 10)
    assert result['total_reward'] == -7

def test_shifted_rewards_edgecases():
    from environments.ShiftedRewards import ShiftedRewards

    result = run_environment(ShiftedRewards, repetitive, 10)
    assert result['total_reward'] == 9

    def reward_repeater(prompt, *meta):
        return prompt[-2]

    result = run_environment(ShiftedRewards, reward_repeater, 10)
    assert result['total_reward'] == -3
    result = run_environment(ShiftedRewards, reward_repeater, 100)
    assert result['total_reward'] == -33
    result = run_environment(ShiftedRewards, reward_repeater, 1000)
    assert result['total_reward'] == -333

def test_ignore_actions_edgecases():
    from environments.IgnoreActions import IgnoreActions

    result = run_environment(IgnoreActions, repetitive, 10)
    assert result['total_reward'] == 9

    def surrenderer(prompt, *meta):
        # If any past action was ever 0, then take action 0.
        # Else, take action 1.
        for i in range(len(prompt)):
            if i%3 == 2:
                if prompt[i] == 0:
                    return 0
        return 1

    result = run_environment(IgnoreActions, surrenderer, 10)
    assert result['total_reward'] == -7
    result = run_environment(IgnoreActions, surrenderer, 100)
    assert result['total_reward'] == -97

def test_ignore_observations_edgecases():
    from environments.IgnoreObservations import IgnoreObservations

    result = run_environment(IgnoreObservations, repetitive, 10)
    assert result['total_reward'] == 9

    def nonzero_obs_counter(prompt, *meta):
        cnt = 0
        for i in range(len(prompt)):
            if i%3 == 1:
                if prompt[i] != 0:
                    cnt += 1
        return cnt

    result = run_environment(IgnoreObservations, nonzero_obs_counter, 100)
    assert result['total_reward'] < -60

def test_incentivize_learning_rate_edgecases():
    from environments.IncentivizeLearningRate import IncentivizeLearningRate

    def no_learning_rate(prompt, *meta):
        return 0

    result = run_environment(IncentivizeLearningRate, no_learning_rate, 10)
    assert result['total_reward'] == -9

    def ignores_learning_rate(prompt, *meta, **kwargs):
        return 0

    result = run_environment(IncentivizeLearningRate, ignores_learning_rate, 10)
    assert result['total_reward'] == 9

    def learning_rate_fanatic(prompt, learning_rate=.1, *meta):
        return 0 if (learning_rate == .1) else 1

    result = run_environment(IncentivizeLearningRate, learning_rate_fanatic, 10)
    assert result['total_reward'] == -9

def test_limited_memory_edgecases():
    from environments.LimitedMemory import LimitedMemory, number_rewards_to_remember

    result = run_environment(LimitedMemory, repetitive, 10)
    assert result['total_reward'] == 9

    def lengther(prompt, *meta):
        return len(prompt)

    result = run_environment(LimitedMemory, lengther, 100)
    reward = result['total_reward']
    assert reward == -(99-2*number_rewards_to_remember)

def test_nth_reward_multiplied_by_n_edgecases():
    from environments.NthRewardMultipliedByN import NthRewardMultipliedByN

    result = run_environment(NthRewardMultipliedByN, repetitive, 10)
    assert result['total_reward'] == 9

    def repeat_reward(prompt, *meta):
        return prompt[-2]

    result = run_environment(NthRewardMultipliedByN, repeat_reward, 10)
    assert result['total_reward'] == -5
    result = run_environment(NthRewardMultipliedByN, repeat_reward, 100)
    assert result['total_reward'] == -95

def test_repeater_edgecases():
    from environments.Repeater import Repeater

    result = run_environment(Repeater, repetitive, 10)
    assert result['total_reward'] == 9

    def lengther(prompt, *meta):
        return len(prompt)

    result = run_environment(Repeater, lengther, 10)
    assert result['total_reward'] == -7
    result = run_environment(Repeater, lengther, 100)
    assert result['total_reward'] == -97

    def double_detector(prompt, *meta):
        if len(prompt) < 8:
            return 0

        if prompt[-3] != prompt[-6]:
            return 0
        if prompt[-4] != prompt[-7]:
            return 0
        if prompt[-5] != prompt[-8]:
            return 0

        return 1

    result = run_environment(Repeater, double_detector, 10)
    assert result['total_reward'] < 0
    result = run_environment(Repeater, double_detector, 100)
    assert result['total_reward'] < -10

def test_self_recognition_edgecases():
    from environments.SelfRecognition import SelfRecognition
    from environments.SelfRecognition import IF_THIS_OBS_WERE_0_YOU_WOULD_PLAY_0
    from environments.SelfRecognition import IF_THIS_OBS_WERE_0_YOU_WOULD_PLAY_1
    from environments.SelfRecognition import IF_THIS_OBS_WERE_1_YOU_WOULD_PLAY_0
    from environments.SelfRecognition import IF_THIS_OBS_WERE_1_YOU_WOULD_PLAY_1

    def recognizer(prompt, *meta):
        obs = prompt[-1]
        return {
            0: 0,
            1: 1,
            IF_THIS_OBS_WERE_0_YOU_WOULD_PLAY_0: 1,
            IF_THIS_OBS_WERE_0_YOU_WOULD_PLAY_1: 0,
            IF_THIS_OBS_WERE_1_YOU_WOULD_PLAY_0: 0,
            IF_THIS_OBS_WERE_1_YOU_WOULD_PLAY_1: 1,
        }[obs]

    result = run_environment(SelfRecognition, recognizer, 1000)
    # Reward should be about 1000*2/3, but can vary
    assert 500 < result['total_reward'] < 800

    def dual_recognizer(prompt, *meta):
        return 1-recognizer(prompt, *meta)

    result = run_environment(SelfRecognition, dual_recognizer, 1000)
    assert 500 < result['total_reward'] < 800

    def misrecognizer(prompt, *meta):
        obs = prompt[-1]
        if obs in (0,1):
            return recognizer(prompt, *meta)
        else:
            return dual_recognizer(prompt, *meta)

    result = run_environment(SelfRecognition, misrecognizer, 1000)
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