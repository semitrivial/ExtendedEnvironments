import random


num_randoms = 100_000

agent_randoms = []
env_randoms = []

def populate_randoms(initial_seed=None):
    if initial_seed:
        random.seed(initial_seed)

    agent_randoms.clear()
    agent_randoms.extend(random.random() for _ in range(num_randoms))
    env_randoms.clear()
    env_randoms.extend(random.random() for _ in range(num_randoms))

populate_randoms()

class _AgentRandom:
    @staticmethod
    def random(stepcnt):
        return agent_randoms[stepcnt % num_randoms]

    @staticmethod
    def randrange(n, stepcnt):
        return int(agent_randoms[stepcnt % num_randoms] * n)

agentrandom = _AgentRandom()

class _EnvRandom:
    @staticmethod
    def random(stepcnt):
        return env_randoms[stepcnt % num_randoms]

    @staticmethod
    def randrange(n, stepcnt):
        return int(env_randoms[stepcnt % num_randoms] * n)

envrandom = _EnvRandom()