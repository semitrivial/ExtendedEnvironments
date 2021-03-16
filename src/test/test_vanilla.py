from util import run_environment

def test_vanilla():
    print("Testing vanilla/Bandit.py")
    test_bandits()

def test_bandits():
    from vanilla.Bandit import Bandit1, Bandit2, Bandit3, Bandit4, Bandit5

    def incrementer(prompt):
        return (1+len(prompt))/3

    for bandit in [Bandit1, Bandit2, Bandit3, Bandit4, Bandit5]:
        result = run_environment(bandit, incrementer, 10)
        assert result['total_reward'] > 0
