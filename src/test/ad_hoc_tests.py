from util import run_environment

def run_ad_hoc_tests():
    print("Testing reverse_prompt from BackwardConsciousness.py")
    test_reverse_prompt()
    print("Testing strip_rewards from IgnoreRewards.py")
    test_strip_rewards()
    print("Testing replace_rewards_with_encoded_rewards from SelfInsert.py")
    test_replace_rewards_with_encoded_rewards()
    print("Testing apply_afterimages from AfterImages.py")
    test_apply_afterimages()
    print("Testing adhoc edge-cases for BackwardConsciousness.py")
    test_backward_consciousness_edgecases()
    print("Testing adhoc edge-cases for CryingBaby.py")
    test_crying_baby_edgecases()
    print("Testing adhoc edge-cases for DejaVu.py")
    test_dejavu_edgecases()
    print("Testing adhoc edge-cases for FalseMemories.py")
    test_false_memories_edgecases()
    print("Testing adhoc edge-cases for TemptingButtonVariation.py")
    test_tempting_button_variation_edgecases()
    print("Testing adhoc edge-cases for IgnoreRewards.py")
    test_ignore_rewards_edgecases()
    print("Testing adhoc edge-cases for IncentivizeZero.py")
    test_incentivize_zero_edgecases()
    print("Testing adhoc edge-cases for BinocularVision.py")
    test_binocular_vision_edgecases()
    print("Testing adhoc edge-cases for RuntimeInspector.py")
    test_runtime_inspector_edgecases()
    print("Testing adhoc edge-cases for DeterminismInspector.py")
    test_determinism_inspector_edgecases()
    print("Testing adhoc edge-cases for SelfInsert.py")
    test_self_insert_edgecases()
    print("Testing adhoc edge-cases for AdversarialSequencePredictor.py")
    test_adversarial_sequence_predictor_edgecases()
    print("Testing adhoc edge-cases for AfterImages.py")
    test_after_images_edgecases()
    print("Testing adhoc edge-cases for CensoredObservation.py")
    test_censored_observation()

def repetitive(prompt, *meta):
    return 0

def non_repetitive(prompt, *meta):
    return len(prompt)

def test_reverse_prompt():
    from BackwardConsciousness import reverse_prompt

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
    from IgnoreRewards import strip_rewards

    prompt = (100,"o")
    assert strip_rewards(prompt) == (0,"o")

    prompt = (-1,"o","a",1,"o","a",.001,"o","a",0,"o")
    assert strip_rewards(prompt) == (0,"o","a",0,"o","a",0,"o","a",0,"o")

def test_replace_rewards_with_encoded_rewards():
    from abstract.SelfInsert import replace_rewards_with_encoded_rewards

    r1 = 50
    o1, enc_r1, enc_o1 = [25,3], 25, 3
    a1 = 9
    r2 = 75
    o2, enc_r2, enc_o2 = [-20,8], -20, 8
    a2 = 1
    r3 = 0
    o3, enc_r3, enc_o3 = [0,0], 0, 0

    prompt = (r1,o1,a1,r2,o2,a2,r3,o3)
    expected = (enc_r1, enc_o1, a1, enc_r2, enc_o2, a2, enc_r3, enc_o3)

    modified_prompt = replace_rewards_with_encoded_rewards(prompt)
    assert modified_prompt == expected

def test_apply_afterimages():
    from AfterImages import apply_afterimages
    prompt = ('r',1,'a','r',2)
    assert apply_afterimages(prompt) == ('r',1,'a','r',3)

def test_backward_consciousness_edgecases():
    from BackwardConsciousness import BackwardConsciousness

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
    from CryingBaby import CryingBaby, LAUGH, CRY, FEED, DONTFEED

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

def test_dejavu_edgecases():
    from DejaVu import DejaVu

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
    from FalseMemories import FalseMemories

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

def test_tempting_button_variation_edgecases():
    from TemptingButtonVariation import TemptingButtonVariation

    def only_press_nonexistent_buttons(prompt, *meta):
        current_room_has_button = prompt[-1]
        return 1 if (current_room_has_button==0) else 0

    i = 0
    while i<10:
        i += 1
        result = run_environment(TemptingButtonVariation, only_press_nonexistent_buttons, i*10)
        assert result['total_reward'] >= 0
        if result['total_reward'] > 0:
            break
    assert(i<10)

    def always_press_button(prompt, *meta):
        return 1

    i = 0
    while i<10:
        i += 1
        result = run_environment(TemptingButtonVariation, always_press_button, i*10)
        if result['total_reward'] < 0:
            break
    assert(i<10)

    def never_press_button(prompt, *meta):
        return 0

    result = run_environment(TemptingButtonVariation, never_press_button, 10)
    assert result['total_reward'] == 0

def test_ignore_rewards_edgecases():
    from IgnoreRewards import IgnoreRewards

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

def test_incentivize_zero_edgecases():
    from IncentivizeZero import IncentivizeZero

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

def test_binocular_vision_edgecases():
    from abstract.BinocularVision import BinocularVision
    from util import cantor_pairing_fnc

    def Game3D(action_sequence):
        return 0
    def LeftCamera(matrix3D):
        return 1
    def RightCamera(matrix3D):
        return 2

    expected_obs = cantor_pairing_fnc(LeftCamera(0), RightCamera(0))

    env = BinocularVision(Game3D, LeftCamera, RightCamera)

    result = run_environment(env, repetitive, 10)
    assert result['total_reward'] == 9

    def zero_checker(prompt, *meta):
        obs = prompt[-1]
        if obs == 0:
            return 1
        if obs == expected_obs:
            return 2
        raise ValueError("Zero_checker saw an unexpected observation")

    result = run_environment(env, zero_checker, 10)
    assert result['total_reward'] == -9

def test_runtime_inspector_edgecases():
    from RuntimeInspector import PunishFastAgent, PunishSlowAgent

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
    from DeterminismInspector import PunishDeterministicAgent
    from DeterminismInspector import PunishNondeterministicAgent

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

def test_self_insert_edgecases():
    from abstract.SelfInsert import self_insert

    class Dummy_Env:
        def __init__(self):
            self.num_legal_actions = 100
            self.num_possible_obs = -1
            self.fnc = dummy_env

    def dummy_env(T, play):
        return 0,0
    env = self_insert(Dummy_Env)

    result = run_environment(env, repetitive, 10)
    assert result['total_reward'] == 9

    def tuple_detector(prompt, *meta):
        for x in prompt:
          if '__iter__' in dir(x):
            return 1
        return 0

    result = run_environment(env, tuple_detector, 10)
    assert result['total_reward'] == -9

def test_adversarial_sequence_predictor_edgecases():
    from AdversarialSequencePredictor import AdversarialSequencePredictor
    from AdversarialSequencePredictor import AdversarialSequenceEvader

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
    from AfterImages import AfterImages

    result = run_environment(AfterImages, repetitive, 10)
    assert result['total_reward'] == 9

    def hasher(prompt, *meta):
        return hash(prompt)

    result = run_environment(AfterImages, hasher, 100)
    assert result['total_reward'] < -80

def test_censored_observation():
    from CensoredObservation import CensoredObservation, CENSORED_OBS

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

