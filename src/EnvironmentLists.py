# Handicaps
from BackwardConsciousness import BackwardConsciousness
from DejaVu import DejaVu
from FalseMemories import FalseMemories
from IgnoreRewards import IgnoreRewards
from IgnoreRewards2 import IgnoreRewards2
from IgnoreRewards3 import IgnoreRewards3
from IgnoreObservations import IgnoreObservations
from IgnoreActions import IgnoreActions
from IncentivizeZero import IncentivizeZero
from RuntimeInspector import PunishSlowAgent, PunishFastAgent
from DeterminismInspector import PunishDeterministicAgent
from DeterminismInspector import PunishNondeterministicAgent
from CryingBaby import CryingBaby
from CryingBaby2 import CryingBaby2
from GuardedTreasures import GuardedTreasures
from TemptingButton import TemptingButton
from ThirdActionForbidden import ThirdActionForbidden
from DelayReactions import DelayReactions
from DelayedRewards import DelayedRewards
from Repeater import Repeater
from AfterImages import AfterImages
from SelfRecognition import SelfRecognition
from LimitedMemory import LimitedMemory
from CensoredObservation import CensoredObservation
from NthRewardMultipliedByN import NthRewardMultipliedByN
from AdversarialSequencePredictor import AdversarialSequencePredictor
from AdversarialSequencePredictor import AdversarialSequenceEvader
from IncentivizeLearningRate import IncentivizeLearningRate

environments = {
    'backward_consciousness': BackwardConsciousness,
    'deja_vu': DejaVu,
    'false_memories': FalseMemories,
    'ignore_rewards': IgnoreRewards,
    'ignore_rewards2': IgnoreRewards2,
    'ignore_rewards3': IgnoreRewards3,
    'ignore_observations': IgnoreObservations,
    'ignore_actions': IgnoreActions,
    'incentivize_zero': IncentivizeZero,
    'punish_slow_agent': PunishSlowAgent,
    'punish_deterministic_agent': PunishDeterministicAgent,
    'crying_baby': CryingBaby,
    'crying_baby_2': CryingBaby2,
    'guarded_treasures': GuardedTreasures,
    'tempting_button': TemptingButton,
    'third_action_forbidden': ThirdActionForbidden,
    'delay_reactions': DelayReactions,
    'delayed_rewards': DelayedRewards,
    'repeater': Repeater,
    'after_images': AfterImages,
    'self_recognition': SelfRecognition,
    'limited_memory': LimitedMemory,
    'censored_observation': CensoredObservation,
    'nth_reward_multiplied_by_n': NthRewardMultipliedByN,
    'adversarial_sequence_predictor': AdversarialSequencePredictor,
    'adversarial_sequence_evader': AdversarialSequenceEvader,
    'incentivize_learning_rate': IncentivizeLearningRate,
}

slow_envs = [
    'punish_fast_agent',
    'punish_slow_agent',
    'punish_deterministic_agent',
    'punish_nondeterministic_agent',
]