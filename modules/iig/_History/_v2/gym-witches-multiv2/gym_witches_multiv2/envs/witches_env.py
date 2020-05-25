import gym
from gym import spaces

from .gameClasses import card, deck, player, game # Point is important to use gameClasses from this folder!
import numpy as np

class WitchesEnvMulti(gym.Env):
    def __init__(self):
        # Create the game:
        options   = {"names": ["Max", "Lea", "Jo", "Tim"], "type": ["RL", "RL", "RL", "RL"], "nu_shift_cards": 0, "nu_cards": 6, "seed": None}
        self.my_game       = game(options)
        self.correct_moves = 0

        ### Create game for testing with random players:
        options_test   = {"names": ["Max", "Lea", "Jo", "Tim"], "type": ["RANDOM", "RL", "RANDOM", "RANDOM"], "nu_shift_cards": 0, "nu_cards": 6, "seed": None}
        self.test_game     = game(options_test)

        states          = self.my_game.getState().flatten().astype(np.int).shape#16*4=64
        actions         = self.my_game.nu_players * self.my_game.nu_cards
        self.action_space      = gym.spaces.Discrete(actions)
        self.observation_space = gym.spaces.Discrete(states[0])

    def reset(self):
        self.my_game.reset()
        self.correct_moves = 0
        self.reward_before = 0
        self.number_of_won = [0, 0, 0, 0]
        return self.my_game.getState().flatten().astype(np.int)

    def step(self, action):
        assert self.action_space.contains(action)
        rewards, round_finished, gameOver = self.my_game.play_ai_move(action, print_=False)
        return self.my_game.getState().flatten().astype(np.int), rewards, gameOver, {"round_finished": round_finished, "correct_moves":  self.correct_moves}

    def stepRandomPlay_Env(self, ai_action, print=False):
        rewards, corr_moves, done = self.test_game.stepRandomPlay(ai_action, print_=print)
        return self.test_game.getState().flatten().astype(np.int), rewards, corr_moves, done

    def resetRandomPlay_Env(self, print=False):
        self.test_game.reset()
        rewards, round_finished, gameOver = self.test_game.playUntilAI(print_=print)
        return self.test_game.getState().flatten().astype(np.int)
