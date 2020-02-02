import numpy as np
from VanilaMCTS import VanilaMCTS
from gameClasses import card, deck, player, game
import datetime
import stdout

# Fall 1:
# nur  2 Spieler, jeder kennt alle Karten
total_rewards  = np.zeros((2,))
num_games      = 10
start_time = datetime.datetime.now()
for i in range(0, num_games):
	game_end = False
	my_game  = game(["Tim", "Bob"])
	state = (my_game.getGameState())
	current_player = my_game.active_player
	mcts_player    = 0
	while not game_end:
		if current_player == mcts_player:
			mcts = VanilaMCTS(n_iterations=1500, depth=15, exploration_constant=300, state=state, player=current_player, game=my_game)
			stdout.disable()
			best_action, best_q, depth = mcts.solve()
			stdout.enable()

		# take action and get game info
		my_game.setState(state+[current_player])
		print("On the table is:", my_game.on_table_cards)
		print(my_game.players[my_game.active_player].hand)
		if current_player == mcts_player:
			action = best_action
			print("<<<<<AI player chose best action!>>>>>")
		else:
			print("Your Choice?")
			#input
			#action = int(input("Action index to play\n"))
			action = my_game.getRandomOption_()

		print("I play:", my_game.players[my_game.active_player].hand[action])
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
	del my_game


# Stats with 12 cards of 2,13,14 of Blue Green Red Yellow and 2 Players
	# Tested with 50 Games, mcts player (always starts) and a random player
	# mcts adjustements: n_iterations=1500, depth=15, exploration_constant=300
	# Result: total_rewards: [-211. -439.] for game 50 total time: 0:06:52.199611
	# rewards per game: -4.22, 8,8 per game
	#
	# Play 10 games with a human opponent
	# rewards: total_rewards: [-64. -66.] for game 10 total time: 0:10:25.083755
	# human lost a close match
	# per game: -6.4, -6.6 per game

# Stats with 60 cards 1-14 15=Joker, and 2 Players
	#(n_iterations=1500, depth=15, exploration_constant=300
