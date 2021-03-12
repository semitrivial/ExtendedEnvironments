def run_ad_hoc_tests():
    print("Testing reverse_prompt from BackwardConsciousness.py")
    test_reverse_prompt()

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