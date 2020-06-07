import gym
from gym import spaces

from .gameClassesTest import card, deck, player, game # Point is important to use gameClasses from this folder!
import numpy as np
import json

class WitchesEnvTest(gym.Env):
    def __init__(self):
        self.action_space      = gym.spaces.Discrete(60)
        self.observation_space = gym.spaces.Discrete(303)

        # Create the game:
        self.use_shifting  = True

        # Create the game:
        self.options = {}
        self.options_file_path =  "gym-witches-multi/gym_witches_multi/envs/reinforce_options.json"
        with open(self.options_file_path) as json_file:
            self.options = json.load(json_file)
        self.reinfo_index = self.options["type"].index("REINFO")
        self.my_game      = game(self.options)

    def reset(self):
        #print("Reset game")
        self.my_game.reset_game()
        self.playUnitlAI()
        state = self.my_game.getState()
        return state.flatten().astype(np.int)

    def step(self, action):
        # SET self.use_shifting = True
        assert self.action_space.contains(action)
        # now has to play AI!!!line
        rewards, round_finished, gameOver = self.play_ai_move(action)
        done = False
        if rewards is None:
            current_player          =  self.my_game.active_player
            rewards                 =  [0, 0, 0, 0]
            rewards[current_player] = -1000
            return None, None, True, {"rewards": rewards, "round_finished": round_finished, "gameOver": gameOver}
        else:
            rewards, round_finished, gameOver = self.playUnitlAI()
            return None, None, gameOver, {"rewards": rewards, "round_finished": round_finished, "gameOver": gameOver}

    def play_ai_move(self, ai_action):
        ' ai_action is an absolute 0....60 action index! (no hand card index)'
        current_player =  self.my_game.active_player
        if "REINFO"  in self.my_game.ai_player[current_player]:
            valid_options_idx = self.my_game.getValidOptions(current_player)
            card   = self.my_game.players[current_player].getIndexOfCard(ai_action)
            player_has_card = self.my_game.players[current_player].hasSpecificCardOnHand(card)
            tmp = self.my_game.players[current_player].specificIndexHand(card)
            if player_has_card and tmp in valid_options_idx:
                # if self.use_shifting and self.my_game.shifting_phase:
                #     print("[{}] {} {}\t shifts {}\tCard {}\tHand Index {}\t len {}".format(self.my_game.current_round, current_player, self.my_game.names_player[current_player], self.my_game.ai_player[current_player], card, ai_action, len(self.my_game.players[current_player].hand)))
                # else:
                #     print("[{}] {} {}\t{}\tCard {}\tHand Index {}\t len {}".format(self.my_game.current_round, current_player, self.my_game.names_player[current_player], self.my_game.ai_player[current_player], card, ai_action, len(self.my_game.players[current_player].hand)))
                return self.my_game.step_idx_with_shift(tmp)
            else:
                return None, None, None
        else:
            print("!!!!!!!!ERROR!!!!!!!!!!!!!!! ai should play now!!")
            return None

    def playUnitlAI(self):
        #print("\nPlay until AI")
        rewards = np.zeros((self.my_game.nu_players,))
        game_over = False
        while len(self.my_game.players[self.my_game.active_player].hand) > 0:
            current_player = self.my_game.active_player
            if "RANDOM" in self.my_game.ai_player[current_player]:
                if self.use_shifting and self.my_game.shifting_phase:
                    action = self.my_game.getRandomCards()[0]
                else:
                    action = self.my_game.getRandomOption_()
                card   = self.my_game.players[current_player].hand[action]
                #print("use_shifting:", self.use_shifting,  "my_game-shifting_phase:", self.my_game.shifting_phase, self.my_game.shifted_cards)
                # if self.use_shifting and self.my_game.shifting_phase:
                #     print("[{}] {} {}\t shifts {}\tCard {}\tHand Index {}\t len {}".format(self.my_game.current_round, current_player, self.my_game.names_player[current_player], self.my_game.ai_player[current_player], card, action, len(self.my_game.players[current_player].hand)))
                # else:
                #     print("[{}] {} {}\t{}\tCard {}\tHand Index {}\t len {}".format(self.my_game.current_round, current_player, self.my_game.names_player[current_player], self.my_game.ai_player[current_player], card, action, len(self.my_game.players[current_player].hand)))
                rewards, round_finished, gameOver = self.my_game.step_idx_with_shift(action)
            else:
                return rewards, True, False
        # Game is over!
        return rewards, round_finished, True
