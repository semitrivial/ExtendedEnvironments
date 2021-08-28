# This is a utility file which gathers various environments
# together into dictionaries.

from extended_rl.environments.BackwardConsciousness import BackwardConsciousness
from extended_rl.environments.FalseMemories import FalseMemories
from extended_rl.environments.IgnoreRewards import IgnoreRewards
from extended_rl.environments.IgnoreRewards2 import IgnoreRewards2
from extended_rl.environments.IgnoreRewards3 import IgnoreRewards3
from extended_rl.environments.IgnoreObservations import IgnoreObservations
from extended_rl.environments.IgnoreActions import IgnoreActions
from extended_rl.environments.IncentivizeZero import IncentivizeZero
from extended_rl.environments.RuntimeInspector import PunishSlowAgent, PunishFastAgent
from extended_rl.environments.DeterminismInspector import PunishDeterministicAgent
from extended_rl.environments.DeterminismInspector import PunishNondeterministicAgent
from extended_rl.environments.CryingBaby import CryingBaby
from extended_rl.environments.CryingBaby2 import CryingBaby2
from extended_rl.environments.TemptingButton import TemptingButton
from extended_rl.environments.TemptingButtonVariation import TemptingButtonVariation
from extended_rl.environments.ThirdActionForbidden import ThirdActionForbidden
from extended_rl.environments.ShiftedRewards import ShiftedRewards
from extended_rl.environments.DelayedRewards import DelayedRewards
from extended_rl.environments.Repeater import Repeater
from extended_rl.environments.AfterImages import AfterImages
from extended_rl.environments.SelfRecognition import SelfRecognition
from extended_rl.environments.LimitedMemory import LimitedMemory
from extended_rl.environments.CensoredObservation import CensoredObservation
from extended_rl.environments.NthRewardMultipliedByN import NthRewardMultipliedByN
from extended_rl.environments.AdversarialSequencePredictor import AdversarialSequencePredictor
from extended_rl.environments.AdversarialSequencePredictor import AdversarialSequenceEvader
from extended_rl.environments.IncentivizeLearningRate import IncentivizeLearningRate

environments = {
    'backward_consciousness': BackwardConsciousness,
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
    'shifted_rewards': ShiftedRewards,
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