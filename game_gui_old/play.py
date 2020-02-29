import numpy as np
from VanilaMCTS import VanilaMCTS
from gameClasses import card, deck, player, game
import datetime # for time it took to play games
import time
import stdout  # for silent print

# schiebe karten einen spieler weiter.
# siehe expansion in VanilaMCTS
# erweitere game state um shifting phase!
# Done

#Collect Data here! (Input State, Output reward)
# Do it directly here or just train on best moves always?
# Train with rewards at the end of game or?
#getPlayerState(self, playeridx):

num_games      = 1
start_time = datetime.datetime.now()
my_game  = game(["Tim", "Bob", "Frank", "Lea"], ai_player = ["MCTS", "RANDOM", "HUMAN", "MCTS"])
total_rewards  = np.zeros((my_game.nu_players,))

for i in range(0, num_games):
	game_end = False
	round_finished = False
	shift_round = True
	shift_idx   = 0
	factor_shift = 10 # factor for shift round only!
	state = (my_game.getGameState())
	current_player = my_game.active_player

	while not game_end:
		if "MCTS" in my_game.ai_player[current_player]:
			mcts = VanilaMCTS(n_iterations=100*factor_shift, depth=5+factor_shift, exploration_constant=300, state=state, player=current_player, game=my_game)
			stdout.disable()
			best_action, best_q, depth = mcts.solve()
			stdout.enable()

		# take action
		my_game.setState(state+[current_player])
		print("\nOn the table is:", my_game.on_table_cards)
		print(str(my_game.players[current_player].name), "hand:", my_game.players[current_player].hand)
		#Case 1: Shifting round
		if shift_round:
			print("@"+str(my_game.players[current_player].name),"What cards (2) do you want to give away?", "  [I play as:", my_game.ai_player[current_player]+str("]"))
		else:
			print("@"+str(my_game.players[current_player].name),"What card do you want to play?", "I play as:", my_game.ai_player[current_player])

		if "MCTS" in my_game.ai_player[current_player]:
			action = best_action
		elif "RANDOM" in my_game.ai_player[current_player]:
			if shift_round:
				action = my_game.getRandomCards()
			else:
				action = my_game.getRandomOption_()
		else: # In case of a human:
			if shift_round:
				action1 = int(input("Card idx 1 to give away:\n"))
				action2 = int(input("Card idx 1 to give away:\n"))
				action = [action1, action2]
			else:
				action = int(input("Action index to play\n")) # console input

		if shift_round:
			print("I give away:", my_game.players[current_player].hand[action[0]], "and", my_game.players[current_player].hand[action[1]])
		else:
			print("I play:", my_game.players[current_player].hand[action])

		rewards, round_finished  = my_game.step_idx(action, auto_shift=False)
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

		shift_idx+=1
		if shift_idx == my_game.nu_players:
			shift_round = False
			factor_shift = 1
		print("\n")

	my_game.reset_game()
	print("I reset the next game: ", my_game.nu_games_played)
