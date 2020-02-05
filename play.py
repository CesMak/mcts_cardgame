import numpy as np
from VanilaMCTS import VanilaMCTS
from gameClasses import card, deck, player, game
import datetime
import stdout

num_games      = 5
start_time = datetime.datetime.now()
my_game  = game(["Tim", "Bob", "Frank", "Ann"], ai_player = ["MCTS", "RANDOM", "MCTS", "RANDOM"])
total_rewards  = np.zeros((my_game.nu_players,))

for i in range(0, num_games):
	game_end = False
	state = (my_game.getGameState())
	current_player = my_game.active_player
	while not game_end:
		if "MCTS" in my_game.ai_player[current_player]:
			mcts = VanilaMCTS(n_iterations=100, depth=6, exploration_constant=300, state=state, player=current_player, game=my_game)
			stdout.disable()
			best_action, best_q, depth = mcts.solve()
			stdout.enable()

		# take action and get game info
		my_game.setState(state+[current_player])
		print("On the table is:", my_game.on_table_cards)
		print(str(my_game.players[current_player].name), "hand:", my_game.players[current_player].hand)
		print("@"+str(my_game.players[current_player].name),"What card do you want to play?", "I play as:", my_game.ai_player[current_player])
		if "MCTS" in my_game.ai_player[current_player]:
			action = best_action
		elif "RANDOM" in my_game.ai_player[current_player]:
			action = my_game.getRandomOption_()
		else: # In case of a human:
			action = int(input("Action index to play\n"))

		print("I play:", my_game.players[current_player].hand[action])
		rewards  = my_game.step_idx(action)
		if rewards is not None:
			print("\nGame finished with offhand:")
			for player in my_game.players:
				print(player.name, "\n", player.offhand)
			print("rewards:", rewards)
			total_rewards += rewards
			#stdout.enable()
			print("total_rewards:", total_rewards, "for game", i+1, "total time:", datetime.datetime.now()-start_time)
			game_end = True

		state = (my_game.getGameState())
		current_player = my_game.active_player
		print("\n")
	my_game.reset_game()
	print("I reset the next game: ", my_game.nu_games_played)
