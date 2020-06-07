import gym
from gym import spaces

from .gameClasses import card, deck, player, game # Point is important to use gameClasses from this folder!
import numpy as np

class WitchesEnvMulti(gym.Env):
    def __init__(self):
        # Create the train game:
        # Lowest 10 cards, if nu_shift_cards=0 no cards are shifted.
        #Max cards in witches is 15
        options   = {"names": ["Max", "Lea", "Jo", "Tim"], "type": ["RL", "RL", "RL", "RL"], "nu_shift_cards": 2, "nu_cards": 15, "seed": None}
        self.my_game       = game(options)
        self.correct_moves = 0

        ### Create the test game (only one RL)
        self.options_test   = {"names": ["Max", "Lea", "Jo", "Tim"], "type": ["RANDOM", "RL", "RANDOM", "RANDOM"], "nu_shift_cards": 2, "nu_cards": 15, "seed": None}
        self.test_game     = game(self.options_test)

        states          = self.my_game.getState().flatten().astype(np.int).shape
        actions         = self.my_game.nu_players * self.my_game.nu_cards
        self.action_space      = gym.spaces.Discrete(actions)
        self.observation_space = gym.spaces.Discrete(states[0])

        # Reward style:
        self.style = "final" # "final"
        self.printON = False

    def reset(self):
        'used in train game'
        self.my_game.reset()
        self.correct_moves = 0
        return self.my_game.getState().flatten().astype(np.int)

    def step(self, action):
        '''
        used in train game
        returns reward according to style
        '''
        assert self.action_space.contains(action)
        rewards, round_finished, done = self.my_game.play_ai_move(action, print_=self.printON)
        rewardss, done = self.selectReward(rewards, round_finished, done, self.style)
        if done:
            state = self.reset()
        else:
            state = self.my_game.getState().flatten().astype(np.int)
        return state, rewardss, done, {"round_finished": round_finished, "correct_moves":  self.correct_moves}

    def selectReward(self, rewards, round_finished, done, style="final"):
        '''
        final:
            - returns -100 for wrong move
            - returns 0    for correct move
            - returns [x, x, x, x] at End of game (Rewards of 4 players)
        '''

        if style == "final":
            if rewards["ai_reward"] is None: # illegal move
                return -100, True #-100 before
            elif round_finished and done and rewards["state"] == "play" and not rewards["ai_reward"] is None:
                return rewards["final_rewards"], True
            else:
                self.correct_moves += 1
                return 0, False
        elif style == "trickReward":
            if rewards["ai_reward"] is None: # illegal move
                return -100 #-100 before
            elif round_finished and done and rewards["state"] == "play" and not rewards["ai_reward"] is None:
                return rewards["final_rewards"]
            elif rewards["state"] == "play":
                self.correct_moves += 1
                return rewards["trick_rewards"][rewards["player_win_idx"]]
            else:
                # shifting phase:
                self.correct_moves += 1
                return 0
        elif style == "notDone":
            if rewards["ai_reward"] is None: # not illegal move but -100
                return -100, False
            elif round_finished and done and rewards["state"] == "play" and not rewards["ai_reward"] is None:
                return rewards["final_rewards"], True
            else:
                self.correct_moves += 1
                return 0, False
        elif style =="same_shape": # required for baselines
            if rewards["ai_reward"] is None: # illegal move
                return [-100, -100, -100, -100]
            elif round_finished and done and rewards["state"] == "play" and not rewards["ai_reward"] is None:
                return rewards["final_rewards"]
            else:
                self.correct_moves += 1
                return [0, 0, 0, 0]

    def stepRandomPlay_Env(self, ai_action, print__=False):
        'used for test game'
        rewards, corr_moves, done = self.test_game.stepRandomPlay(ai_action, print_=print__)
        return self.test_game.getState().flatten().astype(np.int), rewards, corr_moves, done

    def resetRandomPlay_Env(self, print__=False):
        'used for test game'
        self.test_game.reset()
        #print Hand of RL player:
        if print__:
            for i in range(4): print("Hand of player: ", self.options_test["names"][i], self.test_game.players[i].hand)
        self.test_game.playUntilAI(print_=print__)
        return self.test_game.getState().flatten().astype(np.int)
