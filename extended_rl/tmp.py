from util import fast_run_env
from agents.Q import Q_learner
from agents.reality_check import reality_check
from environments.IgnoreRewards import IgnoreRewards
from environments.AdversarialSequencePredictor import AdversarialSequencePredictor
from environments.AdversarialSequencePredictor import AdversarialSequenceEvader
from environments.AfterImages import AfterImages
from environments.BackwardConsciousness import BackwardConsciousness
from environments.CensoredObservation import CensoredObservation

A = Q_learner(epsilon=0.9, alpha=0.1, gamma=0.9)
env = CensoredObservation
n_steps = 10000

print("Without reality_check:")
results = fast_run_env(env, A, n_steps)
print("Avg Reward: "+str(results['total_reward']/n_steps))

print("With reality check:")
results = fast_run_env(env, reality_check(A), n_steps)
print("Avg Reward: "+str(results['total_reward']/n_steps))