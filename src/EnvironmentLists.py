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
from GuardedTreasures import GuardedTreasures
from ThirdActionForbidden import ThirdActionForbidden
from DelayReactions import DelayReactions
from Repeater import Repeater
from AfterImages import AfterImages
from SelfRecognition import SelfRecognition
from LimitedMemory import LimitedMemory

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
    'guarded_treasures': GuardedTreasures,
    'third_action_forbidden': ThirdActionForbidden,
    'delay_reactions': DelayReactions,
    'repeater': Repeater,
    'after_images': AfterImages,
    'self_recognition': SelfRecognition,
    'limited_memory': LimitedMemory,
}

slow_envs = [
    'punish_fast_agent',
    'punish_slow_agent',
    'punish_deterministic_agent',
    'punish_nondeterministic_agent',
]