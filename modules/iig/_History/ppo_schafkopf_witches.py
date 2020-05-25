import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim.lr_scheduler import  StepLR
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils import data

import gym
import gym_witches_multiv2
import datetime

# For exporting the model:
import torch.onnx
import onnx
import onnxruntime

import numpy as np
import os

# 4. DataSet class
import experience_dataset_lstm
from experience_dataset_lstm import ExperienceDatasetLSTM

## Contains:
# 1. Model Class
# 2. PPO Class
# 3. Memory Class
# 5. WitchesGame Class

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ActorCriticNetworkLinear(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCriticNetworkLinear, self).__init__()

        self.state_dim      = state_dim
        self.action_dim     = action_dim
        self.info_dim       = (state_dim - 4*action_dim)+3*action_dim

        self.hidden_neurons = 256

        self.fc1 = nn.Linear(self.info_dim, self.hidden_neurons)
        self.fc2 = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        self.fc3a = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        self.fc3b = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        self.fc4a = nn.Linear(self.hidden_neurons, self.hidden_neurons)
        self.fc4b = nn.Linear(self.hidden_neurons, 1)

        self.fc5a = nn.Linear(self.hidden_neurons+action_dim, action_dim) #played options size und action_dim

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, in_, multi=False):
        'in_  = state, len_state'
        lll          = self.action_dim
        info_vector    = []
        options_vector = []
        if multi:
            print("inside multi-forward - input state: ", type(in_), in_.shape, "one state len:", lll)
            print(in_.shape[0], lll)
            on_table = torch.zeros(in_.shape[0], lll)
            print(on_table.shape)
            print("hallo")
            print(in_[:][0:lll].shape)
            on_table[:] = in_[:][0:lll]
            on_table[:], on_hand, played, play_options, add_states = in_[:][0:lll], in_[:][lll:2*lll], in_[:][lll*2:lll*3], in_[:][lll*3:lll*4], in_[:][lll*4:]
            print(type(on_table), len(on_table))
            info_vector = torch.cat([on_table, on_hand, played, add_states ]).float()
            play_options = play_options.float()
        else:
            on_table, on_hand, played, play_options, add_states = in_[0:lll], in_[lll:2*lll], in_[lll*2:lll*3], in_[lll*3:lll*4], in_[lll*4:]
            # convert numpy to torch float tensor!
            info_vector  = torch.from_numpy(np.concatenate((on_table, on_hand, played, add_states))).float()
            play_options = torch.from_numpy(play_options).float()

        # ax for the actor  bx is for the critic output
        x = F.relu(self.fc1(info_vector))
        x = F.relu(self.fc2(x))
        ax = F.relu(self.fc3a(x))
        bx = F.relu(self.fc3b(x))
        ax = self.fc4a(ax)
        bx = self.fc4b(bx)

        if len(in_.shape)==1:
            actor_out =torch.cat( [ax, play_options], 0)
        else:
            actor_out =torch.cat( [ax, play_options], 1)

        actor_out = self.fc5a(actor_out)
        actor_out =  F.softmax(actor_out, dim=-1)

        return actor_out, bx

    def evaluate(self, state_vector, action):
        #TODO- nur erstes Element immer wieder rein?!!! Das ist doch falsch?! vorher: self(state_vector[0], multi=True)
        print("Evaluate:", state_vector.shape, action.shape)
        action_probs, state_value = self(state_vector, multi=True) # ruft forward auf
        dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, policy, lr_params, betas, gamma, K_epochs, eps_clip, batch_size, mini_batch_size, c1=0.5, c2=0.01, start_episode = -1):
        [self.lr, self.lr_stepsize, self.lr_gamma] = lr_params

        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.batch_size= batch_size
        self.mini_batch_size = mini_batch_size

        if batch_size % mini_batch_size != 0:
            raise Exception("batch_size needs to be a multiple of mini_batch_size")

        self.c1 = c1
        self.c2 = c2

        self.policy = policy
        self.optimizer = torch.optim.Adam(self.policy.parameters(),lr=self.lr, betas=betas, weight_decay=5e-5)
        #self.lr_scheduler = StepLR(self.optimizer, step_size=self.lr_stepsize, gamma=self.lr_gamma)
        #self.lr_scheduler.step(start_episode)
        self.policy_old = type(policy)(policy.state_dim, policy.action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory, i_episode):

        self.policy.train() #<- was macht das??!

        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            #rewards.insert(0, discounted_reward)
            rewards.append(discounted_reward)
        rewards.reverse()

        # Normalizing the rewards:
        rewards = np.array(rewards)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-5)

        # Create dataset from collected experiences
        #experience_dataset = ExperienceDatasetLinear(memory.states, memory.actions, memory.allowed_actions, memory.logprobs, rewards)
        ### states, actions, logprobs, rewards
        # Works: Memory values are correctly passed to DataSet! (all same length)
        experience_dataset = ExperienceDatasetLSTM(memory.states, memory.actions, memory.logprobs, rewards)

        training_generator = data.DataLoader(experience_dataset, collate_fn=experience_dataset_lstm.custom_collate, batch_size=self.batch_size, shuffle=True)
#        training_generator = data.DataLoader(experience_dataset, collate_fn=experience_dataset.custom_collate, batch_size=self.mini_batch_size, shuffle=True)
        # TODO AttributeError: 'ExperienceDatasetLSTM' object has no attribute 'custom_collate'
        print("before the DataLoader")
        #training_generator = data.DataLoader(experience_dataset,  batch_size=self.mini_batch_size, shuffle=True)

        # Optimize policy for K epochs:
        avg_loss = 0
        avg_value_loss = 0
        avg_entropy = 0
        avg_clip_fraction = 0
        avg_approx_kl_divergence = 0
        avg_explained_var = 0
        count = 0
        for epoch in range(self.K_epochs): #epoch

            mini_batches_in_batch = int(self.batch_size / self.mini_batch_size)
            self.optimizer.zero_grad()

            for i, (old_states, old_actions, old_logprobs, old_rewards) in enumerate(training_generator): # mini batch
                print(i, old_states.shape, old_actions.shape, old_logprobs.shape, old_rewards.shape)

                # Transfer to GPU
                # old_states = [old_state.to(device) for old_state in old_states]
                # old_actions, old_logprobs, old_rewards = old_actions.to(device), old_logprobs.to(device), old_rewards.to(device)

                # Evaluating old actions and values :
                # TODO!!!
                # old_states ist riesig, state_values ist leer.....
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

                # Finding the ratio (pi_theta / pi_theta__old):
                ratios = torch.exp(logprobs - old_logprobs.detach())

                # Finding Surrogate Loss:
                advantages = old_rewards.detach() - state_values.detach()
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
                value_loss = self.MseLoss(state_values, old_rewards)
                loss = -torch.min(surr1, surr2) + self.c1 * value_loss - self.c2 * dist_entropy

                clip_fraction = (abs(ratios - 1.0) > self.eps_clip).type(torch.FloatTensor).mean()
                approx_kl_divergence = .5 * ((logprobs - old_logprobs.detach()) ** 2).mean()
                explained_var = 1-torch.var(old_rewards - state_values) / torch.var(old_rewards)

                #logging losses only in the first epoch, otherwise they will be dependent on the learning rate
                #if epoch == 0:
                avg_loss += loss.mean().item()
                avg_value_loss += value_loss.mean().item()
                avg_entropy += dist_entropy.mean().item()
                avg_clip_fraction += clip_fraction.item()
                avg_approx_kl_divergence += approx_kl_divergence.item()
                avg_explained_var += explained_var.mean().item()
                count+=1

                #loss.mean().backward()

                if (i + 1) % mini_batches_in_batch == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        # self.writer.add_scalar('Loss/policy_loss', avg_loss/count, i_episode)
        # self.writer.add_scalar('Loss/value_loss', avg_value_loss / count, i_episode)
        # self.writer.add_scalar('Loss/entropy', avg_entropy / count, i_episode)
        # self.writer.add_scalar('Loss/learning_rate', self.lr_scheduler.get_lr()[0], i_episode)
        # self.writer.add_scalar('Loss/ppo_clipping_fraction', avg_clip_fraction/count, i_episode)
        # self.writer.add_scalar('Loss/approx_kl_divergence', avg_approx_kl_divergence / count, i_episode)
        # self.writer.add_scalar('Loss/avg_explained_var', avg_explained_var / count, i_episode)

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def append_memory(self, last_state, action, prob, rewards_, done):
        # if list -> game is finished!
        if isinstance(rewards_, int):
            self.rewards.append(rewards_)
        else:
            self.rewards = self.rewards[:len(self.rewards)-3] # for total of 4 players
            self.rewards.extend(torch.from_numpy(rewards_).float())
        self.actions.append(action)
        self.states.append(last_state)
        self.logprobs.append(prob)
        self.is_terminals.append(done)

    def __str__(self):
        out = "----------- states ------------------\n"
        for counter, s in enumerate(self.states):
            out += "----------- state "+str(counter)+"\n"
            if type(s) == list:
                for c in s:
                    out += np.array2string((c.cpu().numpy()), separator=', ', max_line_width=10000) + "\n"
            else:
                out += np.array2string((s.cpu().numpy()), separator=', ', max_line_width=10000) + "\n"

        out += "----------- rewards ------------------\n"
        out += str(self.rewards)
        return out


class WitchesGame:
    """

    """
    def __init__(self):
        ############## Hyperparameters ##############
        update_games = 20000  # update policy every n games
        batch_size = update_games * 6
        mini_batch_size = 20000 # make this as large as possible to fit in gpu

        eval_games = 500
        checkpoint_folder = "../policies"

        #lr = 0.0002
        lr = 0.001
        lr_stepsize = 30000000 #300000
        lr_gamma = 0.3

        betas = (0.9, 0.999)
        gamma = 0.99  # discount factor
        K_epochs = 16 #8  # update policy for K epochs
        eps_clip = 0.2  # clip parameter for PPO
        c1, c2 = 0.5, 0.005#0.001
        random_seed = None #<---------------------------------------------------------------- set to None
        #############################################
        ### Setup memory:
        self.memory   = Memory()

        ### Setup gym:
        env_name      = "Witches_multi-v2"
        env           = gym.make(env_name)
        state_dim     = env.observation_space.n
        action_dim    = env.action_space.n

        ### Create PPO
        model   = ActorCriticNetworkLinear
        policy  = model(state_dim, action_dim).to(device)
        max_gen = 0
        ppo     = PPO(policy, [lr, lr_stepsize, lr_gamma], betas, gamma, K_epochs, eps_clip, batch_size, mini_batch_size, c1=c1, c2=c2, start_episode=max_gen-1  )

        # training loop
        i_episode = max_gen
        for _ in range(0, 100):   # 90000000
            # play a bunch of games
            for _ in range(update_games):  #update_games
                self.playOneGame(env, ppo.policy_old)
                i_episode += 1

            #update the policy
            # ppo.logger.info("Saving Checkpoint")
            # torch.save(ppo.policy_old.state_dict(), checkpoint_folder + "/" + str(i_episode).zfill(8) + ".pt")
            print("update policy")
            ppo.update(self.memory, i_episode)
            #ppo.lr_scheduler.step(i_episode)

            self.playTestGame(env, ppo.policy_old)


    def playOneGame(self, env, policy):
        #play one 4 player game (all have same old policy)
        done = False
        state= env.reset()
        while not done:
            i      = env.my_game.active_player
            # act braucht state, allowed actions
            last_state                 = state
            action, prob               = self.act(policy, state)
            state, reward, done, info  = env.step(action, "final")
            self.memory.append_memory(last_state, action, prob, reward, done)

    def playTestGame(self, env, policy, nu_games=1000):
        corr_moves = 0
        for i in range(0, nu_games):
            state = env.reset()
            done  = False
            while not done:
                action, prob               = self.act(policy, state)
                state, reward, done, info  = env.step(action, "final")
                corr_moves += info["correct_moves"]
        print("corr_moves", corr_moves/nu_games)

    def act(self, policy, state):
        # allowed actions stecken im state drinnen!
        # state  = on_table+ on_hand+ played+ play_options+ add_states
        action_probs, value = policy(state)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_prob = dist.probs[action].item() # only for debugging purposes
        return action.item(), action_prob

if __name__ == '__main__':
    WitchesGame()
