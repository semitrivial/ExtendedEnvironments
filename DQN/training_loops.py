from VanillaAgentDQN import *
import CryingBaby
import GuardedTreasure

def run_guarded_treasure():
	game_env = GuardedTreasure.GuardedTreasure_v2()
	dqn_agent = RecurrentAgent(network=TreasureGRUNet, game_env=game_env, lookback=10) 
	dqn_agent.play(episodes=5_000) 

	return dqn_agent

def run_crying_baby():
	game_env = CB.CryingBaby_v2()
	adult = RecurrentAgent(network=TreasureGRUNet,game_env=game_env,lookback=10)
	baby = RecurrentAgent(network=TreasureGRUNet,game_env=game_env,lookback=10)
	agents = [adult,baby]

	multiagent_training_loop(game_env, agents, episodes=5_000)

	return agents
