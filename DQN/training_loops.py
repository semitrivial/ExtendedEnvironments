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

def run_all_guarded_treasure_extended(low_performance=True):
	agents = []
	handicaps = [
		GuardedTreasure.for_the_worthy_extension,
		GuardedTreasure.ignore_rewards,
		GuardedTreasure.backward_consciousness,
		GuardedTreasure.deja_vu,
		GuardedTreasure.incentivize_zero
	]
	
	for handicap in extensions:
		game_env = GuardedTreasure.GuardedTreasure_v3(extended=True, extension_reward_function=handicap)
		agent = RecurrentAgent(network=TreasureGRUNet,game_env=game_env,lookback=10)
		agent.append(agent)

		if low_performance==True:
			if handicap == GuardedTreasure.incentivize_zero:
				agent.play(episodes = 1000)
				continue

		agent.play(episodes = 5000)

	return agents



