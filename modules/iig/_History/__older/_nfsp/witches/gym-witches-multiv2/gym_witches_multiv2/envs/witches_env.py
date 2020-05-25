import gym
from gym import spaces

from .gameClasses import card, deck, player, game # Point is important to use gameClasses from this folder!
import numpy as np

class WitchesEnvMulti(gym.Env):
    def __init__(self):
        # Create the game:
        options   = {"names": ["Max", "Lea"], "type": ["RL", "RL"], "nu_shift_cards": 0, "nu_cards": 16, "seed": None}
        self.my_game       = game(options)
        self.correct_moves = 0

        ### Create game for testing with random players:
        options_test   = {"names": ["Max", "Lea", "Jo", "Tim"], "type": ["RANDOM", "RL", "RANDOM", "RANDOM"], "nu_shift_cards": 0, "nu_cards": 6, "seed": None}
        self.test_game     = game(options_test)

        states          = self.my_game.getState(player=0).flatten().astype(np.int).shape#16*4=64
        actions         = self.my_game.nu_players * self.my_game.nu_cards
        self.action_space      = gym.spaces.Discrete(actions)
        self.observation_space = gym.spaces.Discrete(states[0])

    def reset(self):
        self.my_game.reset()
        self.correct_moves = 0
        self.reward_before = 0
        self.number_of_won = [0, 0, 0, 0]
        state_players = []
        for i in range(len(self.my_game.names_player)):
            state_players.append(self.my_game.getState(player=i).flatten().astype(np.int))
        return state_players

    def step_onePlayer(self, action, print=False):
        assert self.action_space.contains(action)
        rewards, round_finished, gameOver = self.my_game.play_ai_move(action, print_=print)
        return self.my_game.getState().flatten().astype(np.int), rewards, gameOver, {"round_finished": round_finished, "correct_moves":  self.correct_moves}

    def getRewards(self, done, rewarding_option, r, player):
        if r["ai_reward"] == -100:
            self.correct_moves += 1
            return -1
        if rewarding_option=="zero_sum":
            #win: 1, loose: 0, not done or illegal move: 0
            if done:
                if "final_rewards" in r: # correct moves played until the end:
                    max_idx = np.argwhere(r["final_rewards"] == max(r["final_rewards"]))
                    # wenn alle gleichen wert haben,
                    if len(max_idx[0]) == 2:
                        #print(eee)
                        return 0
                    else:
                        if player == max_idx[0][0]:
                            return 1
                        else:
                            return 0
                else: # illegal move played
                    return 0
            else:
                return 0
        elif rewarding_option=="trick_reward":
            return r["ai_reward"]
        elif rewarding_option=="final_reward":
            #TODO to be implemented!
            return 0

    def step(self, action, print=False):
        'action: {1: 1, 2: 6}, player: card index to play'
        states,rewards = [], []
        done          = False
        player        = 0
        for key,value in action.items():
            s,r,g, _ = self.step_onePlayer(value, print=print)
            states.append(s)
            if g:
                done=True
            rewards.append(self.getRewards(g, "zero_sum", r, player))
            player +=1
        return states,rewards,done, self.correct_moves

    def stepRandomPlay_Env(self, ai_action, print=False):
        rewards, corr_moves, done = self.test_game.stepRandomPlay(ai_action, print_=print)
        return self.test_game.getState().flatten().astype(np.int), rewards, corr_moves, done

    def resetRandomPlay_Env(self, print=False):
        self.test_game.reset()
        rewards, round_finished, gameOver = self.test_game.playUntilAI(print_=print)
        return self.test_game.getState().flatten().astype(np.int)
