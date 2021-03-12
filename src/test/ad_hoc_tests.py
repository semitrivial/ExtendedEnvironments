def run_ad_hoc_tests():
    print("Testing reverse_prompt from BackwardConsciousness.py")
    test_reverse_prompt()
    print("Testing nutrition from CryingBaby.py")
    test_nutrition()

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