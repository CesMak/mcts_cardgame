import gym
from gym import spaces

from .gameClasses import card, deck, player, game # Point is important to use gameClasses from this folder!
import numpy as np

class WitchesEnvMulti(gym.Env):
    def __init__(self):
        self.action_space      = gym.spaces.Discrete(60)
        self.observation_space = gym.spaces.Discrete(303)

        # Create the game:
        self.my_game       = game()
        self.correct_moves = 0
        self.use_shifting  = True

        self.correct_moves = 0

        ### Create game for testing with random players:
        self.test_game     = game()
        self.test_game.init_Random_TestGame()

    def reset(self):
        self.my_game.reset_game()
        self.correct_moves = 0
        self.reward_before = 0
        self.number_of_won = [0, 0, 0, 0]
        return self.my_game.getState().flatten().astype(np.int)

    def step(self, action):
        assert self.action_space.contains(action)
        rewards, round_finished, gameOver = self.play_ai_move(action)
        return self.my_game.getState().flatten().astype(np.int), rewards, gameOver, {"round_finished": round_finished, "correct_moves":  self.correct_moves}

    def play_ai_move(self, ai_action, print_=False):
        ' ai_action is an absolute 0....60 action index! (no hand card index)'
        current_player    =  self.my_game.active_player
        valid_options_idx = self.my_game.getValidOptions(current_player)
        card              = self.my_game.players[current_player].getIndexOfCard(ai_action)
        player_has_card   = self.my_game.players[current_player].hasSpecificCardOnHand(card)
        tmp               = self.my_game.players[current_player].specificIndexHand(card)
        if player_has_card and tmp in valid_options_idx:
            if print_:
                if self.my_game.shifting_phase:
                    print("[{}] {} {}\t shifts {}\tCard {}\tHand Index {}\t len {}".format(self.my_game.current_round, current_player, self.my_game.names_player[current_player], self.my_game.ai_player[current_player], card, ai_action, len(self.my_game.players[current_player].hand)))
                else:
                    print("[{}] {} {}\t{}\tCard {}\tHand Index {}\t len {}".format(self.my_game.current_round, current_player, self.my_game.names_player[current_player], self.my_game.ai_player[current_player], card, ai_action, len(self.my_game.players[current_player].hand)))
            self.correct_moves +=1
            return self.my_game.step_idx_with_shift(tmp)
        else:
            return {"state": "play_or_shift", "ai_reward": None}, False, True # rewards round_finished, game_over



    def stepRandomPlay_Env(self, ai_action, print=False):
        rewards, corr_moves, done = self.test_game.stepRandomPlay(ai_action, print_=print)
        return self.test_game.getState().flatten().astype(np.int), rewards, corr_moves, done

    def resetRandomPlay_Env(self):
        self.test_game.reset_game()
        return self.test_game.getState().flatten().astype(np.int)
