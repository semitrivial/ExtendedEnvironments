# Handicaps
from BackwardConsciousness import BackwardConsciousness
from DejaVu import DejaVu
from FalseMemories import FalseMemories
from IgnoreRewards import IgnoreRewards
from IgnoreObservations import IgnoreObservations
from IgnoreActions import IgnoreActions
from IncentivizeZero import IncentivizeZero
from RuntimeInspector import PunishSlowAgent, PunishFastAgent
from DeterminismInspector import PunishDeterministicAgent
from DeterminismInspector import PunishNondeterministicAgent
from CryingBaby import CryingBaby
from CryingBaby2 import CryingBaby2
from GuardedTreasures import GuardedTreasures
from GuardedTreasures_Eager import GuardedTreasures_Eager
from ThirdActionForbidden import ThirdActionForbidden
from DelayReactions import DelayReactions
from Repeater import Repeater
from AfterImages import AfterImages
from SelfRecognition import SelfRecognition
from LimitedMemory import LimitedMemory
from CensoredObservation import CensoredObservation

environments = {
    'backward_consciousness': BackwardConsciousness,
    'deja_vu': DejaVu,
    'false_memories': FalseMemories,
    'ignore_rewards': IgnoreRewards,
    'ignore_observations': IgnoreObservations,
    'ignore_actions': IgnoreActions,
    'incentivize_zero': IncentivizeZero,
    'punish_slow_agent': PunishSlowAgent,
    'punish_deterministic_agent': PunishDeterministicAgent,
    'crying_baby': CryingBaby,
    'crying_baby_2': CryingBaby2,
    'guarded_treasures': GuardedTreasures,
    'guarded_treasures_eager': GuardedTreasures_Eager,
    'third_action_forbidden': ThirdActionForbidden,
    'delay_reactions': DelayReactions,
    'repeater': Repeater,
    'after_images': AfterImages,
    'self_recognition': SelfRecognition,
    'limited_memory': LimitedMemory,
    'censored_observation': CensoredObservation,
}

slow_envs = [
    'punish_fast_agent',
    'punish_slow_agent',
    'punish_deterministic_agent',
    'punish_nondeterministic_agent',
]