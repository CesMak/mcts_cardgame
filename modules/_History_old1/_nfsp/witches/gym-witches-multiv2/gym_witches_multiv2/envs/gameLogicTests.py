import unittest
from gameClasses import card, deck, player, game
import numpy as np

class gameLogic(unittest.TestCase):

    ## NFSP Tests:

    def test_getState(self):
        print("\ntest_getState:")
        options   = {"names": ["Max", "Lea"], "type": ["RL", "RL"], "nu_shift_cards": 0, "nu_cards": 8, "seed": 111}
        my_game   = game(options)

        for play in my_game.players:
            print("\t", play.hand, len(play.hand))

        #test binary options:
        options = my_game.players[my_game.active_player].getBinaryOptions(my_game.getInColor(), my_game.nu_players, my_game.nu_cards)
        cards = my_game.state2Cards(options)
        print(options, cards)

    # def test_playToEnd(self):
    #     options   = {"names": ["Max", "Lea"], "type": ["RL", "RL"], "nu_shift_cards": 0, "nu_cards": 8, "seed": 111}
    #     my_game   = game(options)
    #     state    = my_game.getState().flatten().astype(np.int).shape#16*4=64
    #     print("Cards:")
    #     for i in range(my_game.nu_players):
    #         print(my_game.names_player[i], my_game.players[i].hand)
    #
    #     # Test 2 play until end (only legal moves):
    #     # {'state': 'play', 'ai_reward': 0, 'on_table_win_idx': 1, 'trick_rewards': [0, 0, 0, 0], 'player_win_idx': 1, 'final_rewards': array([-12.,  -1.])} 16 True
    #     for i in [0, 2, 3, 1, 4, 5, 6, 8, 7, 9, 11, 10, 14, 12, 15, 13]:
    #         rewards, round_finished, gameOver = my_game.play_ai_move(i, print_=True)
    #     print("\n")

    # def test_illegal_move(self):
    #     options   = {"names": ["Max", "Lea"], "type": ["RL", "RL"], "nu_shift_cards": 0, "nu_cards": 8, "seed": 111}
    #     my_game   = game(options)
    #     state    = my_game.getState().flatten().astype(np.int).shape#16*4=64
    #     print("Cards:")
    #     for i in range(my_game.nu_players):
    #         print(my_game.names_player[i], my_game.players[i].hand)
    #
    #     # Test 1 illegal move:
    #     # False {'state': 'play_or_shift', 'ai_reward': -100} True
    #     for i in range(2):
    #         rewards, round_finished, gameOver = my_game.play_ai_move(0, print_=True)
    #         print(round_finished, rewards, gameOver)
    #
    #     print("\n")

    # def test_2players(self):
    #     options   = {"names": ["Max", "Lea"], "type": ["RANDOM", "RANDOM"], "nu_shift_cards": 0, "nu_cards": 4, "seed": None}
    #     my_game   = game(options)
    #     states          = my_game.getState().flatten().astype(np.int).shape#16*4=64
    #     print("\n")

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

    # def test_randomPlay(self):
    #     print("inside test random play")

if __name__ == '__main__':
    unittest.main()
