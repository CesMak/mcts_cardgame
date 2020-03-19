# tested with python 3.7.5
from typing import List

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np

# For exporting the model:
import torch.onnx
import onnx
import onnxruntime
import stdout

from gameClasses import card, deck, player, game
import json
import matplotlib.pyplot as plt
import datetime

# Links:
# use logProb: https://pytorch.org/docs/stable/distributions.html

class PolicyGradientLoss(nn.Module):
    def forward(self, log_action_probabilities, discounted_rewards):
        # log_action_probabilities -> (B, timesteps, 1)
        # discounted_rewards -> (B, timesteps, 1)
        losses = -discounted_rewards * log_action_probabilities # -> (B, timesteps, 1)
        loss = losses.mean()
        print("Mean Loss  :" ,round(loss.item(), 5), "Shape losses:", losses.shape)
        return loss

class WitchesPolicy(nn.Module):
    def __init__(self):
        super(WitchesPolicy, self).__init__()
        self.n_inputs  = 180 # 3*60
        self.n_outputs = 60
        self.lr        = 0.01
        self.network = nn.Sequential(
            nn.Linear(self.n_inputs, 500),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.ReLU(),
            nn.Linear(500, 500),
            nn.ReLU(),
            nn.Linear(500, self.n_outputs),
            nn.Softmax(dim=-1))
        self.optimizer = optim.SGD(self.parameters(), lr=self.lr)
        self.criterion = PolicyGradientLoss() #other class

        self.log_action_probabilities = []
        self.rewards = []

    def forward(self, state: torch.tensor, legalCards: torch.tensor):
        state = state.resize_(180)
        probs = self.network(torch.FloatTensor(state))
        print("\n\n\nPROBS in FORWARD:")
        print(probs)
        probs = probs * legalCards
        distribution = Categorical(probs)               #
        action_idx = distribution.sample()              #
        log_action_probability = distribution.log_prob(action_idx)
        self.log_action_probabilities.append(log_action_probability)
        returned_tensor = torch.zeros(1, 2)
        returned_tensor[:, 0] = action_idx.item()
        returned_tensor[:, 1] = log_action_probability
        return returned_tensor

    def discount_rewards_2(self, rewards, gamma=0.9):
        discountedRewards = list()
        numRewards = len(self.rewards)
        for i in range(numRewards):
            realReward = 0
            for j in range(i, numRewards):
                reward = self.rewards[j]
                realReward += reward * np.power(gamma, j - i)
            discountedRewards.append(realReward)

        rewards = torch.tensor(discountedRewards).unsqueeze(dim=1)
        return rewards

    def discount_rewards(self, rewards, gamma=0.99):
        r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
        # Reverse the array direction for cumsum and then
        # revert back to the original order
        r = r[::-1].cumsum()[::-1]
        result = r - r.mean()
        return  torch.tensor(result).unsqueeze(dim=1)

    def feedback(self, reward: float):
        self.rewards.append(reward)

    def updatePolicy(self):
        log_action_probabilities = torch.stack(self.log_action_probabilities)
        rewards  = torch.tensor(self.rewards) # self.discount_rewards(self.rewards)
        #rewards2 = self.discount_rewards_2(self.rewards)

        print("Rewards  :" ,"%.2f "*len(self.rewards) % tuple(self.rewards))
        print("Action prob:")
        print(log_action_probabilities)
        #print("Disc. Rew:" ,"%.2f "*len(rewards) % tuple(rewards))
        #print("Disc. Rew2:" ,"%.2f "*len(rewards2) % tuple(rewards2))

        # Optimization step
        self.optimizer.zero_grad()
        loss = self.criterion(log_action_probabilities, rewards)
        loss.backward()
        # clipping to prevent nans:
        # see https://discuss.pytorch.org/t/proper-way-to-do-gradient-clipping/191/6
        #torch.nn.utils.clip_grad_norm_(self.parameters(), 5)
        torch.nn.utils.clip_grad_value_(self.parameters(), 5)
        self.optimizer.step()
        self.log_action_probabilities.clear()
        self.rewards.clear()

class PlayingPolicy(nn.Module):
    def __init__(self):
        super().__init__()

        # Parameters
        self.gamma = 0.9 # reward discount factor

        # Network
        in_channels = 4   # four players
        out_channels = 15 # 15 cards per player

        # belot: weight of size 8 5 3 4, expected input[1, 4, 3, 32] to have 5 channels, but got 4 channels instead
        self.conv418 = nn.Conv2d(1, out_channels, kernel_size=(3, 4), stride=1, dilation=(1, 8))
        self.conv881 = nn.Conv2d(1, out_channels, kernel_size=(3, 8), stride=8, dilation=(1, 1))
        self.classify = nn.Linear(645, in_channels*out_channels)

        # Optimizer
        self.optimizer = optim.SGD(self.parameters(), lr=1e-2, momentum=0.9)
        #self.optimizer = optim.Adam(self.parameters(), lr=1e-2)

        # Criterion
        self.criterion = PolicyGradientLoss() #other class

        # Keep track of actions and rewards during a single game (i.e. multiple hands)
        # This will be used to build a batch (torch.cat) passed to the criterion
        self.log_action_probabilities = list()

        # Rewards stored for each round [-0.25, 0.2857142857142857, -0.3392857142857143, 0.375, 0.10714285714285714, -0.35714285714285715, 0.42857142857142855, 0.75]
        # normalizedReward between -1 and 1
        self.rewards = list()

    def forward(self, state: torch.tensor, legalCards: torch.tensor):
        '''
        state: torch.tensor
            - 1x60  Cards  on the table
            - 1x60  Cards hand of current player
            - 1x60  Cards played
        legalCards: toch.tensor
            - 1x60  Cards [0..1...0] which are possible to be played
        RETURN: torch.tensor
        action_idx.item()
        returned_tensor[:, 1] = log_action_probability

        In order to export an onnx model the inputs AND outputs of this function have to be TENSORS
        See also here: https://discuss.pytorch.org/t/torch-onnx-export-fails/72418/2
        '''
        #print("Inside Forward:")
        #print(state)
        #print(state.shape) #[4, 3, 32]

        # state -> 4 (players), 3 (card states), 60 (cards)
        out418 = F.relu(self.conv418(state)) # -> (batch_size, out_channels, ?, ?)
        out881 = F.relu(self.conv881(state)) # -> (batch_size, out_channels, ?, ?)

        #print(out418.shape)# torch.Size([1, 15, 1, 58])
        #print(out881.shape) # torch.Size([1, 15, 1, 58])

        out = torch.cat((
            out418.view(out418.size(0), -1), # convert from 1,8,1,8 to 1,64
            out881.view(out881.size(0), -1),
        ), dim=1) # -> (batch_size, ?)

        #print(out.shape) # print(out.shape)  # torch.Size([1, 1740])

        probs = F.softmax(self.classify(out), dim=1)
        # print(probs.shape)
        # print(probs)
        # print(legalCards.shape)
        # print(legalCards)
        # mask e.g.: 0. 0., 0., 0., 1., 0., 0., 1.,....
        # mask. shape: 1x60,    probs.shape: 1x60
        probs = probs * legalCards # probs.shape = 1x60
        #print(probs)# e.g. tensor([[0.000, 0.0266, 0.0000,...]], grad_fn=<MulBackward0>)

        #print(probs)

        # Get action
        distribution = Categorical(probs) # print(distribution) Categorical(probs: torch.Size([1, 32]))

        action_idx = distribution.sample() # print(action_idx)  # tensor([22])
        # In case it throws this error:
        # Categorical(probs).sample() generates RuntimeError: invalid argument 2: invalid multinomial distribution
        # the softmax had turned into a vector of lovely NaNs. then categorical fails with above error.
        # The model itself is generating the nan because of the exploding gradients due to the learning rate.

        log_action_probability = distribution.log_prob(action_idx) # print(log_action_probability) # tensor([-1.9493], grad_fn=<SqueezeBackward1>)

        # Remember the log-probability
        self.log_action_probabilities.append(log_action_probability)
        #print(log_action_probability)

        returned_tensor = torch.zeros(1, 2)
        returned_tensor[:, 0] = action_idx.item()
        returned_tensor[:, 1] = log_action_probability
        return returned_tensor

    def feedback(self, reward: float):
        self.rewards.append(reward)

    def discount_rewards(self, rewards, gamma=0.99):
        r = np.array([gamma**i * rewards[i] for i in range(len(rewards))])
        # Reverse the array direction for cumsum and then
        # revert back to the original order
        r = r[::-1].cumsum()[::-1]
        result = r - r.mean()
        return  torch.tensor(result).unsqueeze(dim=1)

    def updatePolicy(self):
        # Log-probabilites of performed actions
        # Convert probabs to 15x1 tensor
        log_action_probabilities = torch.cat(self.log_action_probabilities, dim=0)
        rewards = torch.tensor(self.rewards) #self.discount_rewards(self.rewards) #
        # discountedRewards = list()
        # numRewards = len(self.rewards)
        # for i in range(numRewards):
        #     realReward = 0
        #
        #     for j in range(i, numRewards):
        #         reward = self.rewards[j]
        #         realReward += reward * np.power(self.gamma, j - i)
        #
        #     discountedRewards.append(realReward)
        #
        # # or here self.rewards?
        # rewards = torch.tensor(discountedRewards).unsqueeze(dim=1)

        # Optimization step
        self.optimizer.zero_grad()

        #stdout.enable()
        print("log_action_probabilities")
        print(torch.round(log_action_probabilities * 10**2) / (10**2))
        print("self.rewards")
        print([round(x, 2) for x in self.rewards])
        #stdout.disable()

        #TODO use rewards here or self.rewards??????
        loss = self.criterion(log_action_probabilities, rewards)
        loss.backward()
        self.optimizer.step()

        # Reset the log-probabilites and rewards
        self.log_action_probabilities.clear()
        self.rewards.clear()

### TODO GAME HERE
class TestReinforce:
    def __init__(self, parent=None):
        #self.playingPolicy = PlayingPolicy()
        self.witchesPolicy  = WitchesPolicy()
        self.options = {}
        self.options_file_path =  "../data/reinforce_options.json"
        with open(self.options_file_path) as json_file:
            self.options = json.load(json_file)
        self.my_game     = game(self.options)

    def notifyTrick(self, value):
        # der schlimmste wert ist -17 (g10, g5, r1, r2)
        # ausser wenn noch mal2 hinzukommt?! dann ist es wohl 21?!
        #value +=21
        normalizedReward = value / 21 # 21 zuvor sonst 26
        if abs(normalizedReward)>1:
            stdout.enable()
            print(normalizedReward)
            print(eeee)
        #self.playingPolicy.feedback(normalizedReward)
        self.witchesPolicy.feedback(normalizedReward)

    def selectAction(self):
        '''
        the returned action is a hand card index no absolut index!
        '''
        current_player = self.my_game.active_player
        if "RANDOM" in self.my_game.ai_player[current_player]:
            action = self.my_game.getRandomOption_()
        elif "REINFO"  in self.my_game.ai_player[current_player]:
            # get state of active player
            active_player, state, options = self.my_game.getState()
            #print("Options", options)
            #print("State: [Ontable, hand, played]\n", state)

            #torch_tensor = self.playingPolicy(torch.tensor(state).float()   , torch.tensor(options))
            torch_tensor = self.witchesPolicy(torch.tensor(state).float(), torch.tensor(options))
            # absolut action index:
            action_idx   = int(torch_tensor[:, 0])
            log_action_probability = torch_tensor[:, 1]
            card   = self.my_game.players[current_player].getIndexOfCard(action_idx)
            action = self.my_game.players[current_player].specificIndexHand(card)
        return action

    def plotHistory(self, array, ai_index, out_path):
        'input: [[ply1, play2, play3, pay4], ...]'
        x = np.linspace(0, len(array), num=len(array))
        y = array
        plt.xlabel("Time")
        plt.ylabel("Number of games won")
        plt.title("Performance")
        for i in range(len(y[ai_index])):
            z = [pt[i] for pt in y]
            if i == ai_index:
                plt.plot(x, z, label = 'ai player     %s' %i)
            else:
                plt.plot(x, z, label = 'random player %s'%i)
        plt.legend()
        plt.savefig(out_path+"result.png")
        plt.show()

    def exportONNX(self, model, input_vector, path):
        torch_out = torch.onnx._export(model, input_vector, path+".onnx",  export_params=True)

    def play(self):
        number_of_won   = [0, 0, 0, 0]
        gameover_limit  = - 70
        history         = []
        ai_player_index = 0
        nuGames         = 50
        out_path        = "models/rl_policy/"
        stdout.disable()
        stdout.write_file(out_path+"output.txt") # contains all logging!
        start_time   = datetime.datetime.now()
        try:
            for j in range(1, 100000000):
                i=0
                while i<nuGames:
                    action = self.selectAction()
                    current_player = self.my_game.active_player
                    card   = self.my_game.players[current_player].hand[action]
                    print("[{}] {} {}\t{}\tCard {}\tHand Index {}".format(self.my_game.current_round, current_player, self.my_game.names_player[current_player], self.my_game.ai_player[current_player], card, action))
                    rewards, round_finished = self.my_game.step_idx(action, auto_shift=False)
                    if round_finished:
                        # player idx of Reinforce
                        self.notifyTrick(rewards[ai_player_index])
                        print("Update rewards: ", rewards, "\n")
                        if len(self.my_game.players[current_player].hand) == 0: # one game finished
                            print("update policy at end of one game!")
                            #self.playingPolicy.updatePolicy()
                            self.witchesPolicy.updatePolicy()
                            print(self.my_game.total_rewards)
                            if min(self.my_game.total_rewards)<=gameover_limit:
                                winner_idx  = np.where((self.my_game.total_rewards == max(self.my_game.total_rewards)))
                                number_of_won[winner_idx[0][0]] +=1
                                self.my_game.total_rewards = [0, 0, 0, 0]
                                i+=1
                                if i == nuGames:
                                    stdout.enable()
                                    print("Win Stats:", number_of_won, "at game", j*nuGames, "for:", self.witchesPolicy.lr,datetime.datetime.now()-start_time, "\n")
                                    active_player, state, options = self.my_game.getState()
                                    state = torch.tensor(state).float().resize_(180)
                                    path = out_path+str(self.witchesPolicy.lr)+"_"+str(j*nuGames)+"__"+str(number_of_won[ai_player_index])
                                    self.exportONNX(self.witchesPolicy.network, state, path)
                                    history.append(number_of_won)
                                    number_of_won = [0, 0, 0, 0]
                                    stdout.disable()
                            self.my_game.reset_game()
            self.plotHistory(history, ai_player_index, out_path)
        except Exception as e:
            stdout.enable()
            print("ERROR!!!!", e)
            print(number_of_won)
            self.plotHistory(history, ai_player_index, out_path)

if __name__ == "__main__":
    trainer = TestReinforce()
    trainer.play()
