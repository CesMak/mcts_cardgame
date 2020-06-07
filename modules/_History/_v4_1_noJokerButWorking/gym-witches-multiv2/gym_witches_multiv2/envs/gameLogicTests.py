import unittest
from gameClasses import card, deck, player, game
import numpy as np

class gameLogic(unittest.TestCase):

    def test_randomPlay(self):
        test_game     = game({"names": ["Max", "Lea", "Jo", "Tim"], "type": ["RL", "RL", "RL", "RL"], "nu_shift_cards": 2, "nu_cards": 15, "seed": 22})
        test_game.reset()
        print("after reset")

        #print Hand of RL player:
        for i in [3, 12, 7, 13, 10, 3, 1, 0]:
            rewards, corr_moves, done = test_game.step(i, print_=True)
            print(i, rewards)

        # test_game.reset()
        # print(test_game.players[1].hand)
        # test_game.playUntilAI(print_=True)
        # for i in [3, 12, 8, 2, 15, 13]:
        #     rewards, corr_moves, done = test_game.stepRandomPlay(i, print_=True)
        # print(rewards, corr_moves, done)
        #
        # test_game.reset()
        # print(test_game.players[1].hand)
        # test_game.playUntilAI(print_=True)
        # for i in [3, 12, 8, 2, 15, 13]:
        #     print("I try now", i)
        #     rewards, corr_moves, done = test_game.stepRandomPlay(i, print_=True)
        #     print("after stepRandom", rewards, done)
        #     if done:
        #         print("inside done, reset game noew!")
        #         test_game.reset()
        #         test_game.playUntilAI(True)
        #
        # print(rewards, corr_moves, done)


    # def returnResults(rewards, round_finished, gameOver):
    #     if rewards["ai_reward"] is None: # illegal move
    #         return None, self.correct_moves, True
    #     elif gameOver and "final_rewards" in rewards:
    #         # case that ai plays last card:
    #         mean_random = (sum(rewards["final_rewards"])- rewards["final_rewards"][1])/3
    #         print("mean_random:", mean_random, rewards["final_rewards"],  self.correct_moves)
    #         return [rewards["final_rewards"][1], mean_random], self.correct_moves, gameOver
    #
    # def test_randomPlay(self):
    #     test_game     = game(         {"names": ["Max", "Lea", "Jo", "Tim"], "type": ["RANDOM", "RL", "RANDOM", "RANDOM"], "nu_shift_cards": 2, "nu_cards": 8, "seed": 18})
    #
    #     # TODO SHIFTING PHASE included!
    #     for i in [5, 8, 9, 0, 20, 22, 24, 13, 28, 25, 0]:
    #         cp = test_game.active_player
    #         if "RL" in test_game.player_type[cp]:                 # if ai is starting
    #             rewards, round_finished, gameOver = test_game.play_ai_move(i, print_=True)
    #             returnResults(rewards, round_finished, gameOver )
    #         elif "RANDOM" in test_game.player_type[cp]:           # if random is starting
    #             rewards, round_finished, gameOver = test_game.playUntilAI(print_=True)
    #             returnResults(rewards, round_finished, gameOver )
    #             rewards, round_finished, gameOver = test_game.play_ai_move(i, print_=True)
    #             returnResults(rewards, round_finished, gameOver )


    # def test_shifting(self):
    #     options   = {"names": ["Max", "Lea", "Jo", "Tim"], "type": ["RL", "RL", "RL", "RL"], "nu_shift_cards": 2, "nu_cards": 10, "seed": 13}
    #     my_game   = game(options)
    #
    #     #shifting:
    #     for i in [1, 4, 0, 5, 3, 14, 6, 38]:
    #         rewards, round_finished, done = my_game.play_ai_move(i, print_=True)
    #         state = my_game.getState().flatten().astype(np.int)
    #         print("\t",rewards, round_finished, done)
    #         print("\tNext state:")
    #         my_game.printCurrentState()
    #
    #     #playing:
    #     for i in [0, 8, 1, 2, 16, 9]:
    #         rewards, round_finished, done = my_game.play_ai_move(i, print_=True)
    #         state = my_game.getState().flatten().astype(np.int)
    #         print("\t",rewards, round_finished, done)
    #         print("\tNext state:")
    #         my_game.printCurrentState()

    #
    # def test_nuCards(self):
    #     for j in range(16):
    #         options   = {"names": ["Max", "Lea", "Jo", "Tim"], "type": ["RANDOM", "RANDOM", "RL", "RANDOM"], "nu_shift_cards": 0, "nu_cards": j, "seed": None}
    #         my_game   = game(options)
    #         print("NuCards:", j)
    #         for play in my_game.players:
    #             print(play.hand, len(play.hand))
    #             self.assertEqual(len(play.hand), j)
    #     print("\n")
    #
    # def test_idx(self):
    #     options   = {"names": ["Max", "Lea", "Jo", "Tim"], "type": ["RL", "RANDOM", "RL", "RANDOM"], "nu_shift_cards": 0, "nu_cards": 4, "seed": -1}
    #     my_game   = game(options)
    #     for i in range(my_game.nu_cards*my_game.nu_players):
    #         card = my_game.idx2Card(i)
    #         self.assertEqual(card.idx, i)
    #         print(card)
    #     print("\n")
    #
    #
    # def test_step(self):
    #     #TODO add assert here
    #     options   = {"names": ["Max", "Lea", "Jo", "Tim"], "type": ["RL", "RANDOM", "RL", "RANDOM"], "nu_shift_cards": 0, "nu_cards": 4, "seed": 5}
    #     my_game   = game(options)
    #     my_game.reset()
    #     for play in my_game.players:
    #         print(play.hand, len(play.hand))
    #     #RL plays first card:
    #     rewards, round_finished, gameOver = my_game.play_ai_move(my_game.players[0].hand[0].idx, print_=True)
    #
    #     print(rewards, round_finished, gameOver)
    #     print("\n")
    #
    #
    # def test_getValidOptions(self):
    #     print("\ntest_getValidOptions:")
    #     options   = {"names": ["Max", "Lea", "Jo", "Tim"], "type": ["RL", "RL", "RL", "RL"], "nu_shift_cards": 0, "nu_cards": 4, "seed": 5}
    #     my_game   = game(options)
    #     my_game.reset()
    #
    #     for play in my_game.players:
    #         print("\t", play.hand, len(play.hand))
    #
    #     print("\n\t>>>>Start options:")
    #     for i in range(4):
    #         options_start = my_game.getValidOptions(i)
    #         self.assertEqual(options_start, [0, 1, 2, 3])
    #
    #     print("\n\t>>>>playing a color at the start: options:")
    #     rewards, round_finished, gameOver = my_game.play_ai_move(my_game.players[0].hand[1].idx, print_=True)
    #     opt1= my_game.getValidOptions(1)
    #     opt2= my_game.getValidOptions(2)
    #     opt3= my_game.getValidOptions(3) # does not has the color!
    #     self.assertEqual(opt1, [1,3])
    #     self.assertEqual(opt2, [2,3])
    #     self.assertEqual(opt3, [0, 1, 2,3])
    #
    #     print("\n\t>>>>playing a joker at the start: options:")
    #     my_game.reset()
    #     rewards, round_finished, gameOver = my_game.play_ai_move(my_game.players[1].hand[3].idx, print_=True)
    #     for i in [0,2,3]:
    #         options = my_game.getValidOptions(i)
    #         self.assertEqual(options, [0, 1, 2, 3])
    #         print("\t\t", i, options)
    #
    #     print("\n\t>>>>playing a joker and color at the start: options:")
    #     my_game.reset()
    #     rewards, round_finished, gameOver = my_game.play_ai_move(my_game.players[2].hand[2].idx, print_=True)
    #     rewards, round_finished, gameOver = my_game.play_ai_move(my_game.players[3].hand[0].idx, print_=True)
    #     print("Active Player Hand:", my_game.players[my_game.active_player].hand)
    #     opt1= my_game.getValidOptions(0) # has no blue!
    #     opt2= my_game.getValidOptions(1)
    #     opt3= my_game.getValidOptions(2)
    #     self.assertEqual(opt1,  [0, 1, 2,3])
    #     self.assertEqual(opt2, [0,3])
    #     self.assertEqual(opt3, [0])
    #     print("\n")
    #
    # def test_getState(self):
    #     print("\ntest_getState:")
    #     options   = {"names": ["Max", "Lea", "Jo", "Tim"], "type": ["RL", "RL", "RL", "RL"], "nu_shift_cards": 0, "nu_cards": 4, "seed": 5}
    #     my_game   = game(options)
    #     my_game.reset()
    #
    #     for play in my_game.players:
    #         print("\t", play.hand, len(play.hand))
    #     #test binary options:
    #     options = my_game.players[my_game.active_player].getBinaryOptions(my_game.getInColor(), my_game.nu_players, my_game.nu_cards, shifting_phase=my_game.shifting_phase)
    #     cards = my_game.state2Cards(options)
    #     self.assertEqual(str(cards), "[13 of G_5, 12 of R_8, J of R_11, 12 of Y_12]")
    #     print(options, cards)
    #
    #     on_table, on_hand, played = my_game.getmyState(my_game.active_player, my_game.nu_players, my_game.nu_cards)
    #     for i,j in zip([on_table, on_hand, played], ["on_table", "on_hand", "played"]):
    #          print(j, i, my_game.state2Cards(i))
    #
    #     rewards, round_finished, gameOver = my_game.play_ai_move(my_game.players[0].hand[0].idx, print_=True)
    #     rewards, round_finished, gameOver = my_game.play_ai_move(my_game.players[1].hand[0].idx, print_=True)
    #     on_table, on_hand, played = my_game.getmyState(my_game.active_player, my_game.nu_players, my_game.nu_cards)
    #     for i,j in zip([on_table, on_hand, played], ["on_table", "on_hand", "played"]):
    #          print(j, i, my_game.state2Cards(i))
    #
    #     rewards, round_finished, gameOver = my_game.play_ai_move(my_game.players[2].hand[1].idx, print_=True)
    #     rewards, round_finished, gameOver = my_game.play_ai_move(my_game.players[3].hand[2].idx, print_=True)
    #     on_table, on_hand, played = my_game.getmyState(my_game.active_player, my_game.nu_players, my_game.nu_cards)
    #     for i,j in zip([on_table, on_hand, played], ["on_table", "on_hand", "played"]):
    #          print(j, i, my_game.state2Cards(i))
    #
    #     ##Testing additional State:
    #     print("\t\t>>>>Add State testing:")
    #     for play in my_game.players:
    #         print("\t", play.hand, len(play.hand))
    #     rewards, round_finished, gameOver = my_game.play_ai_move(my_game.players[2].hand[0].idx, print_=True)
    #     rewards, round_finished, gameOver = my_game.play_ai_move(my_game.players[3].hand[0].idx, print_=True)
    #     rewards, round_finished, gameOver = my_game.play_ai_move(my_game.players[0].hand[0].idx, print_=True)
    #     for i in range(4):
    #         print("Add state:",i, my_game.getAdditionalState(i), "win idx, free of B G R Y")
    #
    # def test_randomPlay(self):
    #     print("inside test random play")

if __name__ == '__main__':
    unittest.main()
