import gym
from gym import error, spaces, utils
from gym.utils import seeding

from .gameClasses import card, deck, player, game # Point is important to use gameClasses from this folder!
import json
import numpy as np

from gym.utils import seeding

from ppo_witches import PPO
import torch

class WitchesEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.seed()
        self.action_space = gym.spaces.Discrete(60)
        self.observation_space = gym.spaces.Discrete(303)

        # Create the game:
        self.options = {}
        self.number_of_won = [0, 0, 0, 0]
        self.saved_results = np.zeros(4,)
        self.options_file_path =  "../data/reinforce_options.json"
        with open(self.options_file_path) as json_file:
            self.options = json.load(json_file)
        self.reinfo_index = self.options["type"].index("REINFO")
        self.my_game      = game(self.options)
        self.correct_moves = 0
        self.use_shifting  = True

        self.ppo_test = PPO(303, 60, 128, 25*1e-8, (0.9, 0.999), 0.999, 5, 0.01)
        # trained_pre33_    test_-2.8
        path = "ppo_models/PPO_Witches-v0_-11.51600000000002_80.0.pth"
        self.ppo_test.policy_old.load_state_dict(torch.load(path))

    def step_old(self, action):
        assert self.action_space.contains(action)
        # play until ai!
        reward = self.playRound_old(action)
        done = False
        if reward == -100:
            # illegal move, just do not play Card!
            state= self.my_game.getState()
            #print("\nGo out of Step with ai_reward:", reward, "Done:", done)
            self.my_game.total_rewards[self.reinfo_index] -=100
            return state.flatten(), float(reward), True, np.zeros(4,)
        if len(self.my_game.players[self.my_game.active_player].hand) == 0: # game finished
            done = True
        rewards, done = self.playUnitlAI_old()
        state= self.my_game.getState()
        self.updateTotalResult_old()
        return state.flatten(), float(reward)+21, done, self.number_of_won

    def reset_old(self):
        # return nump.nd array 180x1
        # state = on_table, on_hand, played, options for current player!
        #print("\nReset Game")
        self.my_game.reset_game()
        # Play until AI!
        self.playUnitlAI_old()
        state = self.my_game.getState()
        return state.flatten()

    def render(self, mode='human', close=False):
        ...

    def getPathAction(self, x):
        return self.ppo_test.policy_old.act(x.flatten(), None)


    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def playUnitlAI_old(self):
        #print("\nPlay until AI")
        rewards = np.zeros((self.my_game.nu_players,))
        game_over = False
        while len(self.my_game.players[self.my_game.active_player].hand) > 0:
            current_player = self.my_game.active_player
            if not "REINFO" in self.my_game.ai_player[current_player]:
                action = self.my_game.getRandomOption_()
                card   = self.my_game.players[current_player].hand[action]
                #print("[{}] {} {}\t{}\tCard {}\tHand Index {}\t len {}".format(self.my_game.current_round, current_player, self.my_game.names_player[current_player], self.my_game.ai_player[current_player], card, action, len(self.my_game.players[current_player].hand)))
                rewards, round_finished = self.my_game.step_idx(action)
            else:
                return rewards, game_over
        # Game is over!
        return rewards, True

    def finishRound(self, desired_action):
        action = self.selectAction(desired_action)
        if action is None:
            print("Illegal Move")
            return -100
        else:
            print("hallo")

    def playRound_old(self, reinfo_action_idx):
        current_player =  self.my_game.active_player
        round_finished = False
        while not round_finished:
            action = self.selectAction_old(reinfo_action_idx)
            if action is None:
                # illegal move just do not play it!
                return -100
            current_player = self.my_game.active_player
            card   = self.my_game.players[current_player].hand[action]
            #print("[{}] {} {}\t{}\tCard {}\tHand Index {}\t nuCards {}".format(self.my_game.current_round, current_player, self.my_game.names_player[current_player], self.my_game.ai_player[current_player], card, action, len(self.my_game.players[current_player].hand)))
            rewards, round_finished = self.my_game.step_idx(action)
        #print("Rewards", rewards)
        return rewards[self.reinfo_index]

    def selectAction_old(self, reinfo_action_idx):
        '''
        the returned action is a hand card index no absolut index!
        reinfo_action_idx:  0.....60
        return None if ai player does not have card or card is invalid!
        '''
        current_player = self.my_game.active_player
        action = None
        if "RANDOM" in self.my_game.ai_player[current_player]:
            action = self.my_game.getRandomOption_()
        elif "REINFO"  in self.my_game.ai_player[current_player]:
            valid_options_idx = self.my_game.getValidOptions(current_player)
            card   = self.my_game.players[current_player].getIndexOfCard(reinfo_action_idx)
            player_has_card = self.my_game.players[current_player].hasSpecificCardOnHand(card)
            tmp = ""
            if player_has_card:
                for option in valid_options_idx:
                    tmp+=str(self.my_game.players[current_player].hand[option])+ " "
            #print(">>AI wants: {}\t> {}\thas card  {}\toptions {}".format(reinfo_action_idx, card, player_has_card, tmp))
            if player_has_card:
                tmp = self.my_game.players[current_player].specificIndexHand(card)
                if tmp in valid_options_idx:
                    action = tmp
        return action

    def updateTotalResult_old(self):
        gameover_limit = -70
        self.number_of_won =  np.zeros(4,)
        if min(self.my_game.total_rewards)<=gameover_limit:
             winner_idx  = np.where((self.my_game.total_rewards == max(self.my_game.total_rewards)))
             self.number_of_won[winner_idx[0][0]] +=1
             self.saved_results         += self.my_game.total_rewards
             self.my_game.total_rewards = np.zeros(4,)
        #if max(self.number_of_won)>=1:


### other functions:
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
            return state.flatten().astype(np.int), float(reward), True, np.zeros(4,), 0.0
        else:
            #Dieser spieler lernt möglist früh die gelbe 11 zu fassen!
            rewards, done = self.playUnitlAI()
            state         = self.my_game.getState()
            self.updateTotalResult()
            self.correct_moves +=1
            #rewarde nun mit unterschied zu vorher!
            return state.flatten().astype(np.int), float(reward)+21, done, self.number_of_won, self.correct_moves

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
        #print("Reset game")
        self.my_game.reset_game()
        self.playUnitlAI()
        state = self.my_game.getState()
        self.correct_moves = 0
        self.reward_before = 0
        self.number_of_won = [0, 0, 0, 0]
        return state.flatten().astype(np.int)


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
                rewards, round_finished = self.my_game.step_idx_with_shift(action)
            elif "TRAINED" in self.my_game.ai_player[current_player]:
                state = self.my_game.getState()
                action = self.getPathAction(state)
                card   = self.my_game.players[current_player].getIndexOfCard(action)
                tmp    = self.my_game.players[current_player].specificIndexHand(card)
                valid_options_idx = self.my_game.getValidOptions(current_player)
                player_has_card = self.my_game.players[current_player].hasSpecificCardOnHand(card)
                if player_has_card and tmp in valid_options_idx:
                    action = tmp
                else:
                    #print("Wrong MOVE!")
                    action = self.my_game.getRandomOption_()
                rewards, round_finished = self.my_game.step_idx_with_shift(action)
            else:
                return rewards, game_over
        # Game is over!
        return rewards, True

    def updateTotalResult(self):
        gameover_limit = -70
        self.number_of_won =  np.zeros(4,)
        if min(self.my_game.total_rewards)<=gameover_limit:
             winner_idx  = np.where((self.my_game.total_rewards == max(self.my_game.total_rewards)))
             self.number_of_won[winner_idx[0][0]] +=1
             self.saved_results         += self.my_game.total_rewards
             self.my_game.total_rewards = np.zeros(4,)
        #if max(self.number_of_won)>=1:

















# import gym, ray
# from ray.rllib.agents import ppo

#

# class Witches(gym.Env):
#     def __init__(self, env_config):
#         print("Inside InIT WITCHES ENV")
#         self.action_space = gym.spaces.Discrete(60)
#         self.observation_space = gym.spaces.Discrete(180)
#
#         # Create the game:
#         self.options = {}
#         self.options_file_path =  "../data/reinforce_options.json"
#         with open(self.options_file_path) as json_file:
#             self.options = json.load(json_file)
#         self.my_game     = game(self.options)
#
#         print("End of WitchesEnv")
#         # Start the first game
#         self.reset()
#
#     def reset(self):
#         print("INSIDE RESET \n\n")
#         self.my_game.reset_game()
#         active_player, state, options = self.my_game.getState()
#         # convert state to 180x1
#         print(np.ndarray.tolist(state[0][0].flatten()))
#         print(len(np.ndarray.tolist(state[0][0].flatten())))
#         print(state[0][0].flatten().shape) #(180,)
#         return np.ndarray.tolist(state[0][0].flatten())
#
#     def step(self, action):
#         print("INSIDE STEP! \n\n")
#         print(action)
#         assert self.action_space.contains(action)
#         done = 0
#         rewards, round_finished = self._takeAction(action)
#         if len(self.my_game.players[current_player].hand) == 0: # game finished
#             done = 1
#         #play until ai!
#         return None#<obs>, <reward: float>, <done: bool>, <info: dict>
#
#     ## additional custom functions:
#     def _takeAction(self, action):
#         print("inside take action:", action)
#         current_player = self.my_game.active_player
#         card   = self.my_game.players[current_player].hand[action]
#         print("[{}] {} {}\t{}\tCard {}\tHand Index {}".format(self.my_game.current_round, current_player, self.my_game.names_player[current_player], self.my_game.ai_player[current_player], card, action))
#         rewards, round_finished = self.my_game.step_idx(action, auto_shift=False)
#
#     def selectAction(self):
#         '''
#         the returned action is a hand card index no absolut index!
#         '''
#         current_player = self.my_game.active_player
#         if "RANDOM" in self.my_game.ai_player[current_player]:
#             action = self.my_game.getRandomOption_()
#         elif "REINFO"  in self.my_game.ai_player[current_player]:
#             # get state of active player
#             active_player, state, options = self.my_game.getState()
#             #print("Options", options)
#             #print("State: [Ontable, hand, played]\n", state)
#
#             #torch_tensor = self.playingPolicy(torch.tensor(state).float()   , torch.tensor(options))
#             # torch_tensor = self.witchesPolicy(torch.tensor(state).float(), torch.tensor(options))
#             # # absolut action index:
#             # action_idx   = int(torch_tensor[:, 0])
#             log_action_probability = torch_tensor[:, 1]
#             card   = self.my_game.players[current_player].getIndexOfCard(action_idx)
#             action = self.my_game.players[current_player].specificIndexHand(card)
#         return action





# # Links:
# # https://github.com/openai/gym/blob/master/gym/envs/toy_text/blackjack.py
# # https://towardsdatascience.com/creating-a-custom-openai-gym-environment-for-stock-trading-be532be3910e
# # https://ray.readthedocs.io/en/latest/rllib-env.html
# # https://bair.berkeley.edu/blog/2018/12/12/rllib/
# # https://github.com/zmcx16/OpenAI-Gym-Hearts # env. to collect ml data
#
# # Read this for understanding!
# # http://karpathy.github.io/2016/05/31/rl/
# # MultiPlayer LoveLetter game: http://web.mit.edu/xbliang/www/pdf/6867-final-paper.pdf
# # rlCard : uno http://rlcard.org/  # does not work as well!!!
#
# #  https://files.pythonhosted.org/packages/a8/47/7bc688d2c06c1d0fbd388b4e2725028b2792e1f652a28b848462a724c972/ray-0.8.2-cp36-cp36m-manylinux1_x86_64.whl
# #  pip install https://ray-wheels.s3-us-west-2.amazonaws.com/master/38ec2e70524a277d5aea307f6c843065ff982da5/ray-0.8.1-cp36-cp36m-manylinux1_x86_64.whl

# ray.init()
# config = {
#     "lr": 0.01,
#     "num_workers": 0,
# }
# trainer = ppo.PPOTrainer(env=WitchesEnv, config= config)
#
# while True:
#     print(trainer.train())
