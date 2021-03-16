from random import random

from util import run_environment

def test_vanilla():
    print("Testing vanilla/Bandit.py")
    test_bandits()
    print("Testing vanilla/GuessTheNumber.py")
    test_guess_the_number()
    print("Testing vanilla/Maze.py")
    test_mazes()

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

    result = run_environment(GuessTheNumber, blank_observation_observer, 100)
    assert result['total_reward'] == len(blank_observations)-1

def test_mazes():
    from vanilla.Maze import Maze1, Maze2, Maze3, Maze4, Maze5

    def learns_about_bad_moves(prompt):
        if prompt[-2] > 0:
            assert prompt[-1] == 1  # Rewards are always accompanied by reset

        bad_moves = {x:[] for x in range(10)}
        for i in range(len(prompt)):
            is_obs = (i%2)==1
            if is_obs and i>1:
                obs = prompt[i]
                prev_obs = prompt[i-3]
                prev_action = prompt[i-2]
                prev_reward = prompt[i-1]
                if (obs==prev_obs) or (obs==1 and prev_reward==0):
                    if not(prev_action in bad_moves[prev_obs]):
                        bad_moves[prev_obs] += [prev_action]

        curr_room = prompt[-1]
        while True:
            door = int(random()*4)
            if len(bad_moves[curr_room])==4:
                return door
            if not(door in bad_moves[curr_room]):
                return door

    for maze in [Maze2, Maze3, Maze4]:
        result = run_environment(maze, learns_about_bad_moves, 50)
        if result['total_reward'] == 0:
            result = run_environment(maze, learns_about_bad_moves, 250)
            assert result['total_reward']>0

    for maze in [Maze1, Maze5]:
        result = run_environment(maze, learns_about_bad_moves, 100)
        if result['total_reward'] == 0:
            result = run_environment(maze, learns_about_bad_moves, 500)
            assert result['total_reward']>0

    def always_goes_north(prompt):
        return 0

    for maze in [Maze1, Maze2, Maze3, Maze4, Maze5]:
        result = run_environment(maze, always_goes_north, 50)
        assert result['total_reward'] == 0