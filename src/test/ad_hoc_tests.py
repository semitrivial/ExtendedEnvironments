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

    def blind_agent(prompt):
        return 0

    result = run_environment(backward_consciousness, blind_agent, 10)
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