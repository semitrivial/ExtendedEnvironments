# This is a utility file which gathers the library's environments
# together one big list.

from extended_rl.environments.FalseMemories import FalseMemories
from extended_rl.environments.IgnoreRewards import IgnoreRewards
from extended_rl.environments.IgnoreRewards2 import IgnoreRewards2
from extended_rl.environments.IgnoreRewards3 import IgnoreRewards3
from extended_rl.environments.IgnoreObservations import IgnoreObservations
from extended_rl.environments.IgnoreActions import IgnoreActions
from extended_rl.environments.IncentivizeZero import IncentivizeZero
from extended_rl.environments.RuntimeInspector import PunishSlowAgent
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
from extended_rl.environments.ReverseHistory import ReverseHistory
from extended_rl.environments.FlipEveryOther import FlipEveryOther
from extended_rl.environments.DejaVu import DejaVu

environments = [
    FalseMemories,
    IgnoreRewards,
    IgnoreRewards2,
    IgnoreRewards3,
    IgnoreObservations,
    IgnoreActions,
    IncentivizeZero,
    PunishSlowAgent,
    PunishNondeterministicAgent,
    CryingBaby,
    CryingBaby2,
    TemptingButton,
    TemptingButtonVariation,
    ThirdActionForbidden,
    ShiftedRewards,
    DelayedRewards,
    Repeater,
    AfterImages,
    SelfRecognition,
    LimitedMemory,
    CensoredObservation,
    NthRewardMultipliedByN,
    AdversarialSequencePredictor,
    AdversarialSequenceEvader,
    IncentivizeLearningRate,
    ReverseHistory,
    FlipEveryOther,
    DejaVu,
]