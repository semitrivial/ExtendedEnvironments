# Handicaps
from BackwardConsciousness import backward_consciousness
from DejaVu import deja_vu
from FalseMemories import false_memories
from IgnoreRewards import ignore_rewards
from IncentivizeZero import incentivize_zero
from RuntimeInspector import punish_slow_agent, punish_fast_agent
from DeterminismInspector import punish_deterministic_agent
from DeterminismInspector import punish_nondeterministic_agent

handicaps = {
    'backward_consciousness': backward_consciousness,
    'deja_vu': deja_vu,
    'false_memories': false_memories,
    'ignore_rewards': ignore_rewards,
    'incentivize_zero': incentivize_zero,
    'punish_slow_agent': punish_slow_agent,
    'punish_deterministic_agent': punish_deterministic_agent,
}

# Vanilla environments
from vanilla.Bandit import Bandit1, Bandit2, Bandit3, Bandit4, Bandit5
from vanilla.Maze import Maze1, Maze2, Maze3, Maze4, Maze5
from vanilla.GuessTheNumber import GuessTheNumber1
from vanilla.GuessTheNumber import GuessTheNumber2
from vanilla.GuessTheNumber import GuessTheNumber3
from vanilla.PaperRockScissors import PaperRockScissors1
from vanilla.PaperRockScissors import PaperRockScissors2
from vanilla.PaperRockScissors import PaperRockScissors3
from vanilla.TicTacToe import TicTacToe1, TicTacToe2, TicTacToe3

vanillas = {
    'Bandit1': Bandit1,
    'Bandit2': Bandit2,
    'Bandit3': Bandit3,
    'Bandit4': Bandit4,
    'Bandit5': Bandit5,
    'Maze1': Maze1,
    'Maze2': Maze2,
    'Maze3': Maze3,
    'Maze4': Maze4,
    'Maze5': Maze5,
    'GuessTheNumber1': GuessTheNumber1,
    'GuessTheNumber2': GuessTheNumber2,
    'GuessTheNumber3': GuessTheNumber3,
    'PaperRockScissors1': PaperRockScissors1,
    'PaperRockScissors2': PaperRockScissors2,
    'PaperRockScissors3': PaperRockScissors3,
    'TicTacToe1': TicTacToe1,
    'TicTacToe2': TicTacToe2,
    'TicTacToe3': TicTacToe3,
}

# Non-handicap extended environments
from CryingBaby import crying_baby
from GuardedTreasures import guarded_treasures

misc_envs = {
    'CryingBaby': crying_baby,
    'GuardedTreasures': guarded_treasures
}