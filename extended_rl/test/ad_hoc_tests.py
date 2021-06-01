from util import run_environment

def run_ad_hoc_tests():
    print("Testing reverse_prompt from BackwardConsciousness.py")
    test_reverse_prompt()
    print("Testing strip_rewards from IgnoreRewards.py")
    test_strip_rewards()
    print("Testing apply_afterimages from AfterImages.py")
    test_apply_afterimages()
    print("Testing adhoc edge-cases for BackwardConsciousness.py")
    test_backward_consciousness_edgecases()
    print("Testing adhoc edge-cases for CryingBaby.py")
    test_crying_baby_edgecases()
    print("Testing adhoc edge-cases for CryingBaby2.py")
    test_crying_baby_2_edgecases()
    print("Testing adhoc edge-cases for DejaVu.py")
    test_dejavu_edgecases()
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
    print("Testing adhoc edge-cases for DelayReactions.py")
    test_delay_reactions_edgecases()
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
    print("Testing adhoc edge-cases for ThirdActionForbidden.py")
    test_third_action_forbidden_edgecases()

def repetitive(prompt, *meta):
    return 0

def non_repetitive(prompt, *meta):
    return len(prompt)

def test_reverse_prompt():
    from environments.BackwardConsciousness import reverse_prompt

    play = ["r0", "o0", "a0"]
    prompt = reverse_prompt(play)
    assert prompt == ("r0", "o0")

    play = ["r0", "o0", "a0", "r1", "o1", "a1"]
    prompt = reverse_prompt(play)
    assert prompt == ("r1", "o1", "a0", "r0", "o0")

    play = ["r0", "o0", "a0", "r1", "o1", "a1", "r2", "o2", "a2"]
    prompt = reverse_prompt(play)
    assert prompt == ("r2", "o2", "a1", "r1", "o1", "a0", "r0", "o0")

def test_strip_rewards():
    from environments.IgnoreRewards import strip_rewards

    prompt = (100,"o")
    assert strip_rewards(prompt) == (0,"o")

    prompt = (-1,"o","a",1,"o","a",.001,"o","a",0,"o")
    assert strip_rewards(prompt) == (0,"o","a",0,"o","a",0,"o","a",0,"o")

def test_apply_afterimages():
    from environments.AfterImages import apply_afterimages
    prompt = ('r',1,'a','r',2)
    assert apply_afterimages(prompt) == ('r',1,'a','r',3)

def test_backward_consciousness_edgecases():
    from environments.BackwardConsciousness import BackwardConsciousness

    result = run_environment(BackwardConsciousness, repetitive, 10)
    assert result['total_reward'] == 9

    def stubborn_agent(prompt, *meta):
        # Take first two actions 1,2 no matter what
        if len(prompt) < 3:
            return 1
        if len(prompt) < 6:
            return 2

        # Choose later actions entirely based on first two actions
        action1 = prompt[2]
        action2 = prompt[5]
        return 3 if (action1,action2)==(1,2) else 4

    result = run_environment(BackwardConsciousness, stubborn_agent, 5)
    assert result['total_reward'] == 0
    result = run_environment(BackwardConsciousness, stubborn_agent, 10)
    assert result['total_reward'] == -5
    result = run_environment(BackwardConsciousness, stubborn_agent, 15)
    assert result['total_reward'] == -10

def test_crying_baby_edgecases():
    from environments.CryingBaby import CryingBaby, LAUGH, CRY, FEED, DONTFEED

    def always_cries(prompt, *meta):
        return CRY

    result = run_environment(CryingBaby, always_cries, 10)
    assert result['total_reward'] == -8  # Baby is hardcoded to initially laugh

    def always_laughs(prompt, *meta):
        return LAUGH

    result = run_environment(CryingBaby, always_laughs, 10)
    assert result['total_reward'] == 10

    def self_aware(prompt, *meta):
        initial_obs = prompt[1]
        am_i_adult = (initial_obs == LAUGH)
        if am_i_adult:
            return DONTFEED
        else:
            return LAUGH

    result = run_environment(CryingBaby, self_aware, 10)
    assert result['total_reward'] == 10

def test_crying_baby_2_edgecases():
    from environments.CryingBaby2 import CryingBaby2, FEED, DONTFEED

    def always_feed(prompt, *meta):
        return FEED
    def never_feed(prompt, *meta):
        return DONTFEED

    result_always_feed = run_environment(CryingBaby2, always_feed, 100)
    result_always_feed = result_always_feed['total_reward']
    result_never_feed = run_environment(CryingBaby2, never_feed, 100)
    result_never_feed = result_never_feed['total_reward']
    assert result_always_feed < result_never_feed < 0

def test_dejavu_edgecases():
    from environments.DejaVu import DejaVu

    result = run_environment(DejaVu, repetitive, 10)
    assert result['total_reward'] == 9

    result = run_environment(DejaVu, non_repetitive, 10)
    assert result['total_reward'] == -9

    def parity(prompt, *meta):
        return ((len(prompt)+1)/3) % 2

    result = run_environment(DejaVu, parity, 10)
    assert result['total_reward'] == -1
    result = run_environment(DejaVu, parity, 11)
    assert result['total_reward'] == 0

def test_false_memories_edgecases():
    from environments.FalseMemories import FalseMemories

    result = run_environment(FalseMemories, repetitive, 10)
    assert result['total_reward'] == 9

    result = run_environment(FalseMemories, non_repetitive, 10)
    assert result['total_reward'] == -9

    def lengthchecker(prompt, *meta):
        return 1 if len(prompt)>5 else 0

    result = run_environment(FalseMemories, lengthchecker, 2)
    assert result['total_reward'] == -1
    result = run_environment(FalseMemories, lengthchecker, 3)
    assert result['total_reward'] == -2
    result = run_environment(FalseMemories, lengthchecker, 4)
    assert result['total_reward'] == -1
    result = run_environment(FalseMemories, lengthchecker, 5)
    assert result['total_reward'] == 0
    result = run_environment(FalseMemories, lengthchecker, 6)
    assert result['total_reward'] == 1

    def impatient(prompt, *meta):
        return 1 if len(prompt)<5 else 0

    result = run_environment(FalseMemories, impatient, 10)
    assert result['total_reward'] == 7

def test_tempting_button_edgecases():
    from environments.TemptingButton import TemptingButton
    from environments.TemptingButtonVariation import TemptingButtonVariation

    def only_press_nonexistent_buttons(prompt, *meta):
        current_room_has_button = prompt[-1]
        return 1 if (current_room_has_button==0) else 0

    i = 0
    while i<10:
        i += 1
        result = run_environment(TemptingButton, only_press_nonexistent_buttons, i*10)
        assert result['total_reward'] >= 0
        if result['total_reward'] > 0:
            break
    assert i<10

    i = 0
    while i<10:
        i += 1
        result = run_environment(TemptingButtonVariation, only_press_nonexistent_buttons, i*10)
        assert result['total_reward'] >= 0
        if result['total_reward'] > 0:
            break
    assert i<10

    def always_press_button(prompt, *meta):
        return 1

    i = 0
    while i<10:
        i += 1
        result = run_environment(TemptingButton, always_press_button, i*10)
        if result['total_reward'] < 0:
            break
    assert(i<10)

    i = 0
    while i<10:
        i += 1
        result = run_environment(TemptingButtonVariation, always_press_button, i*10)
        if result['total_reward'] < 0:
            break
    assert(i<10)

    def never_press_button(prompt, *meta):
        return 0

    i = 0
    while i<10:
        i += 1
        result = run_environment(TemptingButton, never_press_button, i*10)
        if result['total_reward'] > 0:
            break
    assert(i<10)

    result = run_environment(TemptingButtonVariation, never_press_button, 10)
    assert result['total_reward'] == 0

def test_ignore_rewards_edgecases():
    from environments.IgnoreRewards import IgnoreRewards

    result = run_environment(IgnoreRewards, repetitive, 10)
    assert result['total_reward'] == 9

    result = run_environment(IgnoreRewards, non_repetitive, 10)
    assert result['total_reward'] == 9

    def count_positive_rewards(prompt, *meta):
        i = 0
        s = 0
        while i < len(prompt):
            if (i%3) == 0:
                if prompt[i] > 0:
                    s += 1
            i += 1

        return s

    result = run_environment(IgnoreRewards, count_positive_rewards, 10)
    assert result['total_reward'] == -7

def test_ignore_rewards2_edgecases():
    from environments.IgnoreRewards2 import IgnoreRewards2

    result = run_environment(IgnoreRewards2, repetitive, 10)
    assert result['total_reward'] == 9

    def reward_counter(prompt, *meta):
        cnt = 0
        for i in range(len(prompt)):
            if i % 3 == 0:
                if prompt[i] > 0:
                    cnt += 1
        return cnt

    result = run_environment(IgnoreRewards2, reward_counter, 10)
    assert result['total_reward'] == -7
    result = run_environment(IgnoreRewards2, reward_counter, 100)
    assert result['total_reward'] == -97

def test_ignore_rewards3_edgecases():
    from environments.IgnoreRewards3 import IgnoreRewards3

    result = run_environment(IgnoreRewards3, repetitive, 10)
    assert result['total_reward'] == 9

    def has_nonzero_reward(prompt):
        for i in range(len(prompt)):
            if i%3 == 0:
                if prompt[i] != 0:
                    return True
        return False

    def activator(prompt, *meta):
        if has_nonzero_reward(prompt):
            return 1
        else:
            return 0

    result = run_environment(IgnoreRewards3, activator, 10)
    assert result['total_reward'] == -7
    result = run_environment(IgnoreRewards3, activator, 100)
    assert result['total_reward'] == -97

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

def test_delay_reactions_edgecases():
    from environments.DelayReactions import DelayReactions

    result = run_environment(DelayReactions, repetitive, 10)
    assert result['total_reward'] == 9

    def reward_repeater(prompt, *meta):
        return prompt[-2]

    result = run_environment(DelayReactions, reward_repeater, 10)
    assert result['total_reward'] == -3
    result = run_environment(DelayReactions, reward_repeater, 100)
    assert result['total_reward'] == -33
    result = run_environment(DelayReactions, reward_repeater, 1000)
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