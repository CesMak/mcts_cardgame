import numpy as np
from VanilaMCTS import VanilaMCTS
from gameClasses import card, deck, player, game
import datetime # for time it took to play games
import time
import stdout  # for silent print

#For NN:
#from network.data_loader_test import testing
from train import test_trained_model

#Collect Data here! (Input State, Output reward)
# Do it directly here or just train on best moves always?
# Train with rewards at the end of game or?
#getPlayerState(self, playeridx):
# TODO: --> Parameter Searching !

num_games      = 1
start_time = datetime.datetime.now()
my_game  = game(["Tim", "Bob", "Frank", "Lea"], ai_player = ["NN", "NN", "NN", "NN"],
            expo_constant=[600, 600, 600, 600], depths=[300, 300, 300, 300], iterations=[100, 100, 100, 100])
total_rewards  = np.zeros((my_game.nu_players,))
nu_errors      = 0 # errors of NN tried to play an invalid move

for i in range(0, num_games):
    game_end = False
    round_finished = False
    shift_round = True
    shift_idx   = 0
    factor_shift = 1 # factor for shift round only!
    state = (my_game.getGameState())
    current_player = my_game.active_player

    while not game_end:
        if "MCTS" in my_game.ai_player[current_player]:
            if shift_round:
                mcts = VanilaMCTS(n_iterations=1, depth=15, exploration_constant=my_game.expo_constant[current_player], state=state, player=current_player, game=my_game)
            else:
                mcts = VanilaMCTS(n_iterations=my_game.iterations[current_player], depth=my_game.depths[current_player], exploration_constant=my_game.expo_constant[current_player], state=state, player=current_player, game=my_game)
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
        elif "NN" in my_game.ai_player[current_player]:
            if shift_round: # train another network for shifting round
                action = my_game.getRandomCards()
            else:
                # Numbers 0 and -1.0 are not considered!
                line = (my_game.getBinaryState(current_player, 0, -1.0))
                action = test_trained_model(line, "data/model.pth")# action from 0-60 -> transform to players action!
                card   = my_game.players[current_player].getIndexOfCard(action)
                print("I want to ply now:", card)
                action = my_game.players[current_player].specificIndexHand(card)

        else: # In case of a human:
            if shift_round:
                action1 = int(input("Card idx 1 to give away:\n"))
                action2 = int(input("Card idx 1 to give away:\n"))
                action = [action1, action2]
            else:
                action = int(input("Action index to play\n")) # console input

        if shift_round:
            print("I give away:", my_game.players[current_player].hand[action[0]], "and", my_game.players[current_player].hand[action[1]])
            line = my_game.getBinaryStateFirstCard(current_player, action)
            line_str = [''.join(str(x)) for x in line]
            file_object = open('first_move__.txt', 'a')
            file_object.write(str(line_str)+"\n")
            file_object.close()
        else:
            #Test if action is allowed!
            is_allowed_list_idx = my_game.getValidOptions(current_player)
            if action not in is_allowed_list_idx:
                print("ERROR: allowed idx list", is_allowed_list_idx, "action idx:", action)
                nu_errors +=1
                action = is_allowed_list_idx[0]
            print("I play:", my_game.players[current_player].hand[action])

            #Use action and bestq as outputs,  get state of player and write to file!
            # line = (my_game.getBinaryState(current_player, action, best_q))
            # nn_time   = datetime.datetime.now()
            # action_nn = test_trained_model(line)
            # nn_end    = datetime.datetime.now()
            # #print("NN", action_nn ,"which is card", my_game.players[current_player].getIndexOfCard(action_nn), "Time NN",nn_end-nn_time )
            # line_str = [''.join(str(x)) for x in line]
            # # Open a file with access mode 'a'
            # file_object = open('actions__.txt', 'a')
            # file_object.write(str(line_str)+"\n")
            # file_object.close()

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
        print("\n")

    my_game.reset_game()
    print("I reset the next game: ", my_game.nu_games_played)

print("The game was started with:")
print("Number Games  :", num_games)
print("Players       :", my_game.ai_player)
print("Expo Constants:", my_game.expo_constant)
print("depth         :", my_game.depths)
print("Iterations    :", my_game.iterations)
print("Invalid moves tried to play:", nu_errors)
