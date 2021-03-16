from util import run_environment

def test_vanilla():
    print("Testing vanilla/Bandit.py")
    test_bandits()
    print("Testing vanilla/GuessTheNumber.py")
    test_guess_the_number()

def test_bandits():
    from vanilla.Bandit import Bandit1, Bandit2, Bandit3, Bandit4, Bandit5

    def incrementer(prompt):
        return (1+len(prompt))/3

    for bandit in [Bandit1, Bandit2, Bandit3, Bandit4, Bandit5]:
        result = run_environment(bandit, incrementer, 10)
        assert result['total_reward'] > 0

def test_guess_the_number():
    from vanilla.GuessTheNumber import GuessTheNumber
    blank_observations = []

    def blank_observation_observer(prompt, blank_obs=blank_observations):
        obs = prompt[-1]
        if obs == 0:
            blank_obs += [obs]

        return (((1+len(prompt))/3)%10)+1

    result = run_environment(GuessTheNumber, blank_observation_observer, 30)
    assert result['total_reward'] == len(blank_observations)-1