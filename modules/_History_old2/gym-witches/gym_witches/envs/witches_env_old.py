import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .gameClasses import card, deck, player, game # Point is important to use gameClasses from this folder!
import json
import numpy as np

from gym.utils import seeding
# Takes 4min for one logging (2000 games)
import onnxruntime

# for using path (to speed up)
# Takes 45sec  for one logging (2000 games)
from ppo_witches import PPO
import torch

class WitchesEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.seed()
        self.action_space = gym.spaces.Discrete(60)
        self.observation_space = gym.spaces.MultiBinary(240)

        # Create the game:
        self.options = {}
        self.number_of_won = [0, 0, 0, 0]
        self.saved_results = np.zeros(4,)
        self.correct_moves = 0
        self.options_file_path =  "gym-witches/gym_witches/envs/reinforce_options.json"
        with open(self.options_file_path) as json_file:
            self.options = json.load(json_file)
        self.reinfo_index = self.options["type"].index("REINFO")
        self.my_game      = game(self.options)

        self.use_shifting = True

    def step(self, action):
        # SET self.use_shifting = True
        assert self.action_space.contains(action)
        # now has to play AI!!!line
        reward = self.play_ai_move(action)
        done = False
        if reward == -100:
            # illegal move, just do not play Card!
            state= self.my_game.getState()
            #print("\nGo out of Step with ai_reward:", reward, "Done:", done)
            self.my_game.total_rewards[self.reinfo_index] -=100
            return state.flatten().astype(np.int), -100.0, True, {"number_of_won": self.number_of_won, "correct_moves": self.correct_moves, "total":self.my_game.total_rewards[self.reinfo_index]}
        else:
            rewards, done = self.playUnitlAI()
            state         = self.my_game.getState()
            self.updateTotalResult()
            #result_reward = 0.5 # correct move
            self.correct_moves +=1
            #Is to hard to learn only at end of phase:
            #if done:
            #    result_reward += (self.my_game.total_rewards[self.reinfo_index]+60)/65

            result_reward=(rewards[self.reinfo_index]+21)/26
            return state.flatten().astype(np.int), result_reward, done, {"number_of_won": self.number_of_won, "correct_moves": self.correct_moves, "total":self.my_game.total_rewards[self.reinfo_index]}

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
                rewards, round_finished = self.my_game.step_idx_with_shift(tmp)
                return rewards[self.reinfo_index]
            else:
                return -100
        else:
            print("!!!!!!!!ERROR!!!!!!!!!!!!!!! ai should play now!!")
            return None

    def reset(self):
        self.my_game.reset_game()
        self.playUnitlAI()
        state = self.my_game.getState()
        self.correct_moves = 0
        self.number_of_won = [0, 0, 0, 0]
        return state.flatten().astype(np.int)

    def render(self, mode='human', close=False):
        ...

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def getPathAction(self, x):
        return self.ppo_test.policy_old.act(x.flatten(), None)

    def getOnnxAction(self, path, x):
        '''Input:
        x:      240x1 list binary values
        path    *.onnx (with correct model)'''
        ort_session = onnxruntime.InferenceSession(path)
        ort_inputs  = {ort_session.get_inputs()[0].name: np.asarray(x.flatten(), dtype=np.float32)}
        ort_outs    = ort_session.run(None, ort_inputs)
        max_value = (np.amax(ort_outs))
        result = np.where(ort_outs == np.amax(ort_outs))
        return result[1][0]

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
                # print("use_shifting:", self.use_shifting,  "my_game-shifting_phase:", self.my_game.shifting_phase, self.my_game.shifted_cards)
                # if self.use_shifting and self.my_game.shifting_phase:
                #     print("[{}] {} {}\t shifts {}\tCard {}\tHand Index {}\t len {}".format(self.my_game.current_round, current_player, self.my_game.names_player[current_player], self.my_game.ai_player[current_player], card, action, len(self.my_game.players[current_player].hand)))
                # else:
                #     print("[{}] {} {}\t{}\tCard {}\tHand Index {}\t len {}".format(self.my_game.current_round, current_player, self.my_game.names_player[current_player], self.my_game.ai_player[current_player], card, action, len(self.my_game.players[current_player].hand)))
                rewards, round_finished = self.my_game.step_idx_with_shift(action)
            elif "TRAINED" in self.my_game.ai_player[current_player]:
                action = self.selectAction(0)
                rewards, round_finished = self.my_game.step_idx(action)
            else:
                return rewards, game_over
        # Game is over!
        return rewards, True

    def playRound(self, reinfo_action_idx):
        current_player =  self.my_game.active_player
        round_finished = False
        while not round_finished:
            action = self.selectAction(reinfo_action_idx)
            if action is None:
                #print("This move is illegal")
                return -100
            current_player = self.my_game.active_player
            card   = self.my_game.players[current_player].hand[action]
            # if self.use_shifting and self.my_game.shifting_phase:
            #     print("[{}] {} {}\tshifts {}\tCard {}\tHand Index {}\t len {}".format(self.my_game.current_round, current_player, self.my_game.names_player[current_player], self.my_game.ai_player[current_player], card, action, len(self.my_game.players[current_player].hand)))
            # else:
            #     print("[{}] {} {}\t{}\tCard {}\tHand Index {}\t len {}".format(self.my_game.current_round, current_player, self.my_game.names_player[current_player], self.my_game.ai_player[current_player], card, action, len(self.my_game.players[current_player].hand)))
            rewards, round_finished = self.my_game.step_idx_with_shift(action)
        #print("Rewards", rewards)
        return rewards[self.reinfo_index]

    def selectAction(self, reinfo_action_idx):
        '''
        the returned action is a hand card index no absolut index!
        reinfo_action_idx:  0.....60
        return None if ai player does not have card or card is invalid!
        '''
        current_player = self.my_game.active_player
        action = None
        if "RANDOM" in self.my_game.ai_player[current_player]:
            if self.use_shifting and self.my_game.shifting_phase:
                action = self.my_game.getRandomCards()[0]

            else:
                action = self.my_game.getRandomOption_()
        elif "REINFO"  in self.my_game.ai_player[current_player]:
            valid_options_idx = self.my_game.getValidOptions(current_player)
            card   = self.my_game.players[current_player].getIndexOfCard(reinfo_action_idx)
            player_has_card = self.my_game.players[current_player].hasSpecificCardOnHand(card)
            tmp = ""
            if player_has_card:
                for option in valid_options_idx:
                    tmp+=str(self.my_game.players[current_player].hand[option])+ " "
            # if self.use_shifting  and self.my_game.shifting_phase:
            #     print(">>AI wants to shift: {}\t> {}\thas card  {}\toptions {}".format(reinfo_action_idx, card, player_has_card, tmp))
            # else:
            #     print(">>AI wants to play: {}\t> {}\thas card  {}\toptions {}".format(reinfo_action_idx, card, player_has_card, tmp),"\n")
            if player_has_card:
                tmp = self.my_game.players[current_player].specificIndexHand(card)
                if tmp in valid_options_idx:
                    action = tmp
        elif "TRAINED" in self.my_game.ai_player[current_player]:
            state = self.my_game.getState()
            #action= self.getOnnxAction("gym-witches/gym_witches/envs/test_-2.8.onnx", state)
            action = self.getPathAction(state)
            card   = self.my_game.players[current_player].getIndexOfCard(action)
            tmp = self.my_game.players[current_player].specificIndexHand(card)
            valid_options_idx = self.my_game.getValidOptions(current_player)
            player_has_card = self.my_game.players[current_player].hasSpecificCardOnHand(card)
            if player_has_card and tmp in valid_options_idx:
                action = tmp
            else:
                #print("Wrong MOVE!")
                action = self.my_game.getRandomOption_()
        return action

    def updateTotalResult(self):
        gameover_limit = -70
        self.number_of_won =  np.zeros(4,)
        if min(self.my_game.total_rewards)<=gameover_limit:
             winner_idx  = np.where((self.my_game.total_rewards == max(self.my_game.total_rewards)))
             self.number_of_won[winner_idx[0][0]] +=1
             self.saved_results         += self.my_game.total_rewards
             self.my_game.total_rewards = np.zeros(4,)
        #if max(self.number_of_won)>=1:
