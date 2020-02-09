import numpy as np
from VanilaMCTS import VanilaMCTS
from gameClasses import card, deck, player, game
from gui import GUI
import datetime # for time it took to play games
import time
import stdout  # for silent print

num_games      = 1
start_time = datetime.datetime.now()
my_game  = game(["Tim", "Bob", "Frank", "Lea"], ai_player = ["MCTS", "RANDOM", "MCTS", "RANDOM"])
total_rewards  = np.zeros((my_game.nu_players,))

# Start graphic GUI:
my_gui = GUI(human_player_idx=-1) # set -1 if you want to see all cards!
my_gui.start() # start thread!
my_gui.names = my_game.names_player

for i in range(0, num_games):
	game_end = False
	round_finished = False
	state = (my_game.getGameState())
	current_player = my_game.active_player

	# Deal Cards for graphic gui:
	for i in range(my_game.nu_players):
		my_gui.dealCards(i, my_game.players[i].getHandCardsSorted())

	while not game_end:
		if "MCTS" in my_game.ai_player[current_player]:
			mcts = VanilaMCTS(n_iterations=10, depth=3, exploration_constant=300, state=state, player=current_player, game=my_game)
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
			action = int(input("Action index to play\n")) # console input

		print("I play:", my_game.players[current_player].hand[action])

		# play card graphically:
		my_gui.playCard(my_game.players[current_player].hand[action], player=current_player, round_finished=round_finished)

		rewards, round_finished  = my_game.step_idx(action)
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

		# Used only for GUI: if round finished!
		if round_finished:
			time.sleep(2)
			my_gui.removeInputCards(winner_idx=current_player, results_=rewards)
		else:
			time.sleep(3)
	my_game.reset_game()
	print("I reset the next game: ", my_game.nu_games_played)
