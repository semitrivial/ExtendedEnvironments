# Handicaps
from BackwardConsciousness import BackwardConsciousness
from DejaVu import DejaVu
from FalseMemories import FalseMemories
from IgnoreRewards import IgnoreRewards
from IncentivizeZero import IncentivizeZero
from RuntimeInspector import PunishSlowAgent, PunishFastAgent
from DeterminismInspector import PunishDeterministicAgent
from DeterminismInspector import PunishNondeterministicAgent

handicaps = {
    'backward_consciousness': BackwardConsciousness,
    'deja_vu': DejaVu,
    'false_memories': FalseMemories,
    'ignore_rewards': IgnoreRewards,
    'incentivize_zero': IncentivizeZero,
    'punish_slow_agent': PunishSlowAgent,
    'punish_deterministic_agent': PunishDeterministicAgent,
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
from CryingBaby import CryingBaby
from GuardedTreasures import GuardedTreasures

misc_envs = {
    'CryingBaby': CryingBaby,
    'GuardedTreasures': GuardedTreasures
}

slow_envs = [
    'punish_fast_agent',
    'punish_slow_agent',
    'punish_deterministic_agent',
    'punish_nondeterministic_agent'
    'TicTacToe'
]