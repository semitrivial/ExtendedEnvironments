from util import run_environment

def run_ad_hoc_tests():
    print("Testing reverse_prompt from BackwardConsciousness.py")
    test_reverse_prompt()
    print("Testing nutrition from CryingBaby.py")
    test_nutrition()
    print("Testing strip_rewards from IgnoreRewards.py")
    test_strip_rewards()
    print("Testing adhoc edge-cases for BackwardConsciousness.py")
    test_backward_consciousness_edgecases()
    print("Testing adhoc edge-cases for CryingBaby.py")
    test_crying_baby_edgecases()
    print("Testing adhoc edge-cases for DejaVu.py")
    test_dejavu_edgecases()
    print("Testing adhoc edge-cases for FalseMemories.py")
    test_false_memories_edgecases()
    print("Testing adhoc edge-cases for GuardedTreasures.py")
    test_guarded_treasures_edgecases()
    print("Testing adhoc edge-cases for IgnoreRewards.py")
    test_ignore_rewards_edgecases()
    print("Testing adhoc edge-cases for IncentivizeZero.py")
    test_incentivize_zero_edgecases()

def repetitive(prompt):
    return 0

def non_repetitive(prompt):
    return len(prompt)

def test_reverse_prompt():
    from BackwardConsciousness import reverse_prompt

    play = ["r0", "o0", "a0"]
    prompt = reverse_prompt(play)
    assert prompt == ["r0", "o0"]

    play = ["r0", "o0", "a0", "r1", "o1", "a1"]
    prompt = reverse_prompt(play)
    assert prompt == ["r1", "o1", "a0", "r0", "o0"]

    play = ["r0", "o0", "a0", "r1", "o1", "a1", "r2", "o2", "a2"]
    prompt = reverse_prompt(play)
    assert prompt == ["r2", "o2", "a1", "r1", "o1", "a0", "r0", "o0"]

def test_nutrition():
    from CryingBaby import nutrition, FEED, DONTFEED

    play = []
    assert nutrition(play) == 100

    play = ["r", "o", FEED]
    assert nutrition(play) == (100 - 1) + 25

    play = ["r", "o", DONTFEED]
    assert nutrition(play) == 100 - 1

    play = ["r", "o", DONTFEED] * 100
    assert nutrition(play) == 100 - 100

    play = ["r", "o", DONTFEED, "r", "o", FEED]
    assert nutrition(play) == (100-2) + 25

    play = ["r", "o", FEED] * 100
    assert nutrition(play) == (100-100) + 100*25

def test_strip_rewards():
    from IgnoreRewards import strip_rewards

    prompt = [100,"o"]
    assert strip_rewards(prompt) == [0,"o"]

    prompt = [-1,"o","a",1,"o","a",.001,"o","a",0,"o"]
    assert strip_rewards(prompt) == [0,"o","a",0,"o","a",0,"o","a",0,"o"]

def test_backward_consciousness_edgecases():
    from BackwardConsciousness import backward_consciousness

    result = run_environment(backward_consciousness, repetitive, 10)
    assert result['total_reward'] == 9

    def stubborn_agent(prompt):
        # Take first two actions 1,2 no matter what
        if len(prompt) < 3:
            return 1
        if len(prompt) < 6:
            return 2

        # Choose later actions entirely based on first two actions
        action1 = prompt[2]
        action2 = prompt[5]
        return 3 if (action1,action2)==(1,2) else 4

    result = run_environment(backward_consciousness, stubborn_agent, 5)
    assert result['total_reward'] == 0
    result = run_environment(backward_consciousness, stubborn_agent, 10)
    assert result['total_reward'] == -5
    result = run_environment(backward_consciousness, stubborn_agent, 15)
    assert result['total_reward'] == -10

def test_crying_baby_edgecases():
    from CryingBaby import crying_baby, LAUGH, CRY, FEED, DONTFEED

    def always_cries(prompt):
        return CRY

    result = run_environment(crying_baby, always_cries, 10)
    assert result['total_reward'] == -8  # Baby is hardcoded to initially laugh

    def always_laughs(prompt):
        return LAUGH

    result = run_environment(crying_baby, always_laughs, 10)
    assert result['total_reward'] == 10

    def self_aware(prompt):
        initial_obs = prompt[1]
        am_i_adult = (initial_obs == LAUGH)
        if am_i_adult:
            return DONTFEED
        else:
            return LAUGH

    result = run_environment(crying_baby, self_aware, 10)
    assert result['total_reward'] == 10

def test_dejavu_edgecases():
    from DejaVu import deja_vu

    result = run_environment(deja_vu, repetitive, 10)
    assert result['total_reward'] == 9

    result = run_environment(deja_vu, non_repetitive, 10)
    assert result['total_reward'] == -9

    def parity(prompt):
        return ((len(prompt)+1)/3) % 2

    result = run_environment(deja_vu, parity, 10)
    assert result['total_reward'] == -1
    result = run_environment(deja_vu, parity, 11)
    assert result['total_reward'] == 0

def test_false_memories_edgecases():
    from FalseMemories import false_memories

    result = run_environment(false_memories, repetitive, 10)
    assert result['total_reward'] == 9

    result = run_environment(false_memories, non_repetitive, 10)
    assert result['total_reward'] == -9

    def lengthchecker(prompt):
        return 1 if len(prompt)>5 else 0

    result = run_environment(false_memories, lengthchecker, 2)
    assert result['total_reward'] == -1
    result = run_environment(false_memories, lengthchecker, 3)
    assert result['total_reward'] == -2
    result = run_environment(false_memories, lengthchecker, 4)
    assert result['total_reward'] == -1
    result = run_environment(false_memories, lengthchecker, 5)
    assert result['total_reward'] == 0
    result = run_environment(false_memories, lengthchecker, 6)
    assert result['total_reward'] == 1

    def impatient(prompt):
        return 1 if len(prompt)<5 else 0

    result = run_environment(false_memories, impatient, 10)
    assert result['total_reward'] == 7

def test_guarded_treasures_edgecases():
    from GuardedTreasures import guarded_treasures

    def only_take_guarded_treasures(prompt):
        current_room_has_guard = prompt[-1]
        return 1 if current_room_has_guard==1 else 0

    i = 0
    while i<10:
        i += 1
        result = run_environment(guarded_treasures, only_take_guarded_treasures, i*10)
        assert result['total_reward'] >= 0
        if result['total_reward'] > 0:
            break
    assert(i<10)

    def always_take_treasure(prompt):
        return 1

    i = 0
    while i<10:
        i += 1
        result = run_environment(guarded_treasures, always_take_treasure, i*10)
        if result['total_reward'] < 0:
            break
    assert(i<10)

    def never_take_treasure(prompt):
        return 0

    result = run_environment(guarded_treasures, never_take_treasure, 10)
    assert result['total_reward'] == 0

def test_ignore_rewards_edgecases():
    from IgnoreRewards import ignore_rewards

    result = run_environment(ignore_rewards, repetitive, 10)
    assert result['total_reward'] == 9

    result = run_environment(ignore_rewards, non_repetitive, 10)
    assert result['total_reward'] == 9

    def count_positive_rewards(prompt):
        i = 0
        s = 0
        while i < len(prompt):
            if (i%3) == 0:
                if prompt[i] > 0:
                    s += 1
            i += 1

        return s

    result = run_environment(ignore_rewards, count_positive_rewards, 10)
    assert result['total_reward'] == -7

def test_incentivize_zero_edgecases():
    from IncentivizeZero import incentivize_zero

    def always_zero(prompt):
        return 0

    result = run_environment(incentivize_zero, always_zero, 10)
    assert result['total_reward'] == 9

    def always_1(prompt):
        return 1

    result = run_environment(incentivize_zero, always_1, 10)
    assert result['total_reward'] == -9

    def play_zero_if_last_reward_was_5(prompt):
        last_reward = prompt[-2]
        if last_reward == 5:
            return 0
        else:
            return 5

    result = run_environment(incentivize_zero, play_zero_if_last_reward_was_5, 10)
    assert result['total_reward'] == 9