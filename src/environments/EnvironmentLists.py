from environments.BackwardConsciousness import BackwardConsciousness
from environments.DejaVu import DejaVu
from environments.FalseMemories import FalseMemories
from environments.IgnoreRewards import IgnoreRewards
from environments.IgnoreRewards2 import IgnoreRewards2
from environments.IgnoreRewards3 import IgnoreRewards3
from environments.IgnoreObservations import IgnoreObservations
from environments.IgnoreActions import IgnoreActions
from environments.IncentivizeZero import IncentivizeZero
from environments.RuntimeInspector import PunishSlowAgent, PunishFastAgent
from environments.DeterminismInspector import PunishDeterministicAgent
from environments.DeterminismInspector import PunishNondeterministicAgent
from environments.CryingBaby import CryingBaby
from environments.CryingBaby2 import CryingBaby2
from environments.TemptingButton import TemptingButton
from environments.TemptingButtonVariation import TemptingButtonVariation
from environments.ThirdActionForbidden import ThirdActionForbidden
from environments.DelayReactions import DelayReactions
from environments.DelayedRewards import DelayedRewards
from environments.Repeater import Repeater
from environments.AfterImages import AfterImages
from environments.SelfRecognition import SelfRecognition
from environments.LimitedMemory import LimitedMemory
from environments.CensoredObservation import CensoredObservation
from environments.NthRewardMultipliedByN import NthRewardMultipliedByN
from environments.AdversarialSequencePredictor import AdversarialSequencePredictor
from environments.AdversarialSequencePredictor import AdversarialSequenceEvader
from environments.IncentivizeLearningRate import IncentivizeLearningRate

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
    'tempting_button': TemptingButton,
    'tempting_button_variation': TemptingButtonVariation,
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