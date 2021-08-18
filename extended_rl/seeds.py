import random


n_seeds = 100_000

seeds = []
env_seeds = []

def populate_seeds(initial_seed=None):
    if initial_seed:
        random.seed(initial_seed)

    seeds.clear()
    seeds.extend(random.sample(range(4_000_000_000), n_seeds))
    env_seeds.clear()
    env_seeds.extend(random.sample(range(4_000_000_000), n_seeds))

populate_seeds()