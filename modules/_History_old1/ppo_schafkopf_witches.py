import torch
import torch.nn as nn
from torch.distributions import Categorical
from torch.optim.lr_scheduler import  StepLR
import torch.nn.functional as F

import gym
import gym_witches_multiv2
import datetime

# For exporting the model:
import torch.onnx
import onnx
import onnxruntime

import numpy as np
import os

## Contains:
# 1. Model Class
# 2. PPO Class
# 3. Memory Class
# 4. WitchesGame Class

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
        self.fc4a = nn.Linear(self.hidden_neurons, self.action_dim)
        self.fc4b = nn.Linear(self.hidden_neurons, 1)

        self.fc5a = nn.Linear(self.hidden_neurons+action_dim, action_dim)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, in_):
        'in_  = state, len_state'
        lll          = self.action_dim
        on_table, on_hand, played, play_options, add_states = in_[0:lll], in_[lll:2*lll], in_[lll*2:lll*3], in_[lll*3:lll*4], in_[lll*4:]
        info_vector  = np.concatenate((on_table, on_hand, played, add_states)) # 87x1

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

    def evaluate(self, state_vector, allowed_actions, action):
        action_probs, state_value = self(state_vector, allowed_actions)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def preprocess(self, game_state, player):
        """
        state_size:
        - info_vector: 328
          - game_type: 8
          - game_player: 4
          - first_player: 4
          - current_scores: 4 (divided by 120 for normalization purpose)
          - remaining cards: 32
          - teams: 4 [bits of players are set to 1]
          - played cards by player: 4*32
          - current_trick: 4 * 36

        """

        ego_player = player.id

        #game state
        game_enc = one_hot_games([game_state.game_type])

        game_player_enc = np.zeros(4)
        if game_state.game_player != None:
            game_player_enc[(game_state.game_player-ego_player)%4] = 1

        first_player_enc = np.zeros(4)
        first_player_enc[(game_state.first_player-ego_player)%4] = 1

        team_encoding = np.zeros(4)
        if game_state.get_player_team() != [None]:
            player_team = [(t-ego_player)%4 for t in game_state.get_player_team()]

            if game_state.game_type[1] != 0 and len(player_team) == 1:
                team_encoding[player_team] = 1
            elif game_state.game_type[1] == 0 and len(player_team) == 2:
                team_encoding[player_team] = 1

        played_cards = np.zeros(32*4)
        for p in range(4):
            cards = [game_state.course_of_game[trick][p] for trick in range(8) if game_state.course_of_game[trick][p] != [None, None]]
            enc_cards = one_hot_cards(cards)
            p_id = (p - ego_player) % 4
            played_cards[p_id*32:(p_id+1)*32] = enc_cards


        current_trick_enc = np.zeros(36*4)

        trick = game_state.trick_number
        for card in range(4):
            if game_state.course_of_game[trick][card] == [None, None]:
                continue
            else:
                card_player = game_state.first_player
                if trick != 0:
                    card_player = game_state.trick_owner[trick - 1]
                card_player = (card_player + card) % 4
                card_player_enc = np.zeros(4)
                card_player_enc[(card_player-ego_player)%4] = 1

                card_enc = one_hot_cards([game_state.course_of_game[trick][card]])

                current_trick_enc[card*36:(card+1)*36] = np.concatenate((card_enc, card_player_enc))


        state_vector = np.concatenate((game_enc, game_player_enc, first_player_enc, np.true_divide(game_state.scores, 120), one_hot_cards(player.cards), played_cards,current_trick_enc , team_encoding))

        return [torch.tensor(state_vector).float().to(device=self.device)]

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
        self.optimizer = torch.optim.Adam(self.policy.parameters(),
                                          lr=self.lr, betas=betas, weight_decay=5e-5)
        self.lr_scheduler = StepLR(self.optimizer, step_size=self.lr_stepsize, gamma=self.lr_gamma)
        self.lr_scheduler.step(start_episode)
        self.policy_old = type(policy)(policy.state_dim, policy.action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory, i_episode):

        self.policy.train()

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


        self.logger.info("AVG rewards: "+ str(np.mean(rewards)))
        self.logger.info("STD rewards: " + str(np.std(rewards)))
        # Normalizing the rewards:
        rewards = np.array(rewards)
        rewards = (rewards - np.mean(rewards)) / (np.std(rewards) + 1e-5)



        # Create dataset from collected experiences
        #experience_dataset = ExperienceDatasetLinear(memory.states, memory.actions, memory.allowed_actions, memory.logprobs, rewards)
        experience_dataset = ExperienceDatasetLSTM(memory.states, memory.actions, memory.allowed_actions,
                                                     memory.logprobs, rewards)

        #training_generator = data.DataLoader(experience_dataset, collate_fn=experience_dataset_linear.custom_collate, batch_size=self.batch_size, shuffle=True)
        training_generator = data.DataLoader(experience_dataset, collate_fn=experience_dataset_lstm.custom_collate, batch_size=self.mini_batch_size, shuffle=True)


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

            for i, (old_states, old_actions, old_allowed_actions, old_logprobs, old_rewards) in enumerate(training_generator): # mini batch

                # Transfer to GPU
                old_states = [old_state.to(device) for old_state in old_states]
                old_actions, old_allowed_actions, old_logprobs, old_rewards = old_actions.to(device), old_allowed_actions.to(device), old_logprobs.to(device), old_rewards.to(device)

                # Evaluating old actions and values :
                logprobs, state_values, dist_entropy = self.policy.evaluate(old_states,old_allowed_actions, old_actions)


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

                loss.mean().backward()

                if (i + 1) % mini_batches_in_batch == 0:
                    self.optimizer.step()
                    self.optimizer.zero_grad()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.writer.add_scalar('Loss/policy_loss', avg_loss/count, i_episode)
        self.writer.add_scalar('Loss/value_loss', avg_value_loss / count, i_episode)
        self.writer.add_scalar('Loss/entropy', avg_entropy / count, i_episode)
        self.writer.add_scalar('Loss/learning_rate', self.lr_scheduler.get_lr()[0], i_episode)
        self.writer.add_scalar('Loss/ppo_clipping_fraction', avg_clip_fraction/count, i_episode)
        self.writer.add_scalar('Loss/approx_kl_divergence', avg_approx_kl_divergence / count, i_episode)
        self.writer.add_scalar('Loss/avg_explained_var', avg_explained_var / count, i_episode)

class Memory:
    def __init__(self):
        self.actions = []
        self.allowed_actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.allowed_actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def append_memory(self, last_state, action, prob, reward, done, allowed_actions):
        self.actions.extend(memory.action)
        self.allowed_actions.extend(memory.allowed_actions)
        self.states.extend(memory.last_state)
        self.logprobs.extend(memory.prob)
        self.rewards.extend(memory.reward)
        self.is_terminals.extend(memory.done)

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
        update_games = 50000  # update policy every n games
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
        self.memory   =[Memory(), Memory(), Memory(), Memory()]

        ### Setup gym:
        env_name      = "Witches_multi-v2"
        env           = gym.make(env_name)
        state_dim     = env.observation_space.n
        action_dim    = env.action_space.n
        last_state    = env.reset()

        ### Create PPO
        model   = ActorCriticNetworkLinear
        policy  = model(state_dim, action_dim).to(device)
        max_gen = 0
        ppo     = PPO(policy, [lr, lr_stepsize, lr_gamma], betas, gamma, K_epochs, eps_clip, batch_size, mini_batch_size, c1=c1, c2=c2, start_episode=max_gen-1  )

        # training loop
        i_episode = max_gen
        for _ in range(0, 1):   # 90000000
            # play a bunch of games
            for _ in range(1):  #update_gamess
                last_state   = self.playOneGame(env, ppo.policy_old, last_state)
                i_episode += 1

        #update the policy
        ppo.update(self.memory, i_episode)
        ppo.lr_scheduler.step(i_episode)

    def playOneGame(self, env, policy, state):
        #play one 4 player game (all have same old policy)
        done = False
        while not done:
            i      = env.my_game.active_player
            # act braucht state, allowed actions
            last_state                 = state
            action, prob               = self.act(env, policy, state)
            state, reward, done, info  = env.step(action)
            print(state, last_state)
            self.memory[i].append_memory(last_state, action, prob, reward, done)

    def act(self, env, policy, state):
        # allowed actions stecken im state drinnen!
        # TODO get allowed actions of state!
        # state  = on_table+ on_hand+ played+ play_options+ add_states
        action_probs, value = policy(state)
        print(action_probs, value)
        dist = Categorical(action_probs)
        action = dist.sample()
        action_prob = dist.probs[action].item() # only for debugging purposes
        return action.item(), action_prob

if __name__ == '__main__':
    WitchesGame()
