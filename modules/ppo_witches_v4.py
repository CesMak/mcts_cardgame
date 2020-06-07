import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import gym_witches_multiv2
import datetime

# For exporting the model:
import torch.onnx
import onnx
import onnxruntime

import numpy as np
import os
import random

from copy import deepcopy # used for baches and memory

# use ray for remote / parallel playing games speed up of 40%
import ray #pip install ray[rllib]
ray.init(num_cpus=12)

# Version History:
#Code reference:
#https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

#   _v2     Note that learning 17 of 17 correct moves takes around 15h
#   _v3     included ray for speed up and parallel playing
#           included ray for random playing games
#           corrected minor errors in random playing
#           tested with: update_timestep=300, max_corr=4
#  _v4      Included weight_decay instead of eps in Adam -> not tested if better
#           Corrected bug in gameClasses see line: card.color == "R" and card.value <15 and card.value!=11 and not self.hasRedEleven():
#           Included actorcritic together...
#           Included using decrease lr (with a scheduler)
#           Included to only finished games for update (Problem: 49939 sollte sein: 50000, dont have all same size as update)
#           Test reward=-1000, lr=0.015, 8(cards)+2(shift), test if after all correct learns good moves?
#               update_step = 20k  bis 9.4 corr_moves kein Problem dann schwankt
#               update_step = 80k  -> Memory error!  (watch -n 5 free -m) CAUSE OF RAY WORKERS / RESTART PC!!! -> see also https://github.com/ray-project/ray/issues/8596
#               update_step = 80k
#                   trembles at 9.85
#                   9.92 after 33min (relativ smooth wieder, has learnt to give away high cards and farbe frei machen)
#                   9.94 after 51min trembles more and more!
#                   9.95 after 1:30   bad moves werden vorgezogen.... :(  -> gestoppt
#               update_step = 160k
#                   trembles at 9.938 (20min)
#                   9.17              (30min)
#                   9.929             (47min)
#                   9.94              (1:10)
#                   9.96              (1:17)  ### -> mean_rew of 4965 finished g. ,-2.8735, corr_moves[max:10] ,9.96, mean_rew ,-9.85
#                   9.965             (1:34)  ### -> 4971 finished g. ,-2.8877, corr_moves[max:10] ,9.964, mean_rew ,-8.67,
#                   9.97              (1:50)  ### -> 4974 finished g. ,-2.926, corr_moves[max:10] ,9.97, mean_rew ,-8.11,
#                   9.972             (2:19)  ### -> 4974 finished g. ,-2.8814, corr_moves[max:10] ,9.972, mean_rew ,-8.07,   2:10:55.
#                   9.4971            (3:31)  ### -> es passiert quasi nix mehr... :(  4971 finished g. ,-2.9057, corr_moves[max:10] ,9.971, mean_rew ,-8.69,   3:29:45.677155,playing t=42.04
#           Test reward=-50 lr=0.01, 8(cards)+2(shift), included scaling rewards! rewards/50,  update_step = 160k
#               funktioniert bisher am besten!
#               9.915                 (15min) ### 2880000, mean_rew of 4919 finished g. ,-2.8638, corr_moves[max:10] ,9.915, mean_rew ,-3.63,   0:15:59.922177,playing t=42.33
#               9.92                  (29min) ### 5120000, mean_rew of 4931 finished g. ,-2.8353, corr_moves[max:10] ,9.923, mean_rew ,-3.49,   0:29:06.597057,playing t=41.83
#               9.94                  (40min)
#               9.95                  (50min)
#               9.96                  (1:03)  ### 4969 finished g. ,-2.7927, corr_moves[max:10] ,9.966, mean_rew ,-3.09,
#               9.964                 (1:34)  ### 4966 finished g. ,-2.8621, corr_moves[max:10] ,9.964, mean_rew ,-3.18,   1:34:39.230308,playing t=44.73
#           Test reward=-50 lr=0.01, 10k, 8(cards)+2(shift), included scaling rewards and INCLUDED CORRECT SEQUENCES!!!!
#              Bestes Ergebnis für finished bisher!
#              4635 finished g. ,-2.6183, corr_moves[max:10] ,9.618, mean_rew ,-6.08,  7min
#              4569 finished g. ,-2.5712, corr_moves[max:10] ,9.525, mean_rew ,-6.66,   0:10:52.100171,playing
#              4557 finished g. ,-2.4929, corr_moves[max:10] ,9.557, mean_rew ,-6.7,   0:15:01.081580,
#           Test reward=-50 lr=0.008, 80k, 8(cards)+2(shift), included scaling rewards and INCLUDED CORRECT SEQUENCES!!!!
#              Nochmals eine Verbesserung mit 80k
#              4911 finished g. ,-2.4944, corr_moves[max:10] ,9.907, mean_rew ,-3.34,   0:18:08.535577,playing t=21.57, lr=[0.0038263752]
#               Nach 1:30 passiert nix mehr.... (lr zu klein?)
#              4950 finished g. ,-2.2632, corr_moves[max:10] ,9.957, mean_rew ,-2.74,   1:30:33.902215,playing t=20.17, lr=[0.00010642235717832917]
#              4962 finished g. ,-2.2928, corr_moves[max:10] ,9.964, mean_rew ,-2.66,   7:14:27.167526,playing t=20.53, lr=[1.1800334242024754e-11]
#           Included current lr in logging
#           Included mean of random player in logging (can be compared 1:1 to ai player)
#           NOTE: NORMALIZE REWARDS USING:
#           rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
#           DO NOT USE:::::   rewards = rewards/50.0
#           FIXED BIG BUG for test game playing dass random spieler muessen farbe bekennen
#           FIXED BIG BUG dass take hand passt
#           FIXED BIG BUG dass im random spiel testing richtig abbricht.
#           29.05.2020 included increase batch size over time, not use lr decreasing
#           30.05.2020 included shuffling ...(no DATA Loader used - works everything correcly?)
#           30.05.2020  With increase_batch size suddenly to 0 drops....
#                       mit 256, 128, 64 lernt nur bis 12.7 korrekten Zuege! (minimum -0.38)
#                       auch mit c3 (noch critic layer) und action layer mehr lernt nur bis 12.7
#                       egal was ich gemacht habe hat immer nur bis 12.7 korrekte Zuege gelernt...
#                       trick rewards hat auch nicht geholfen!
#                       batches[jjj].is_terminals.append(True) --> commenting this hilft auch nix!
#                       notDone hat auch nicht geholfen...
#          02.06.2020   Fixed bug in random playing!!!
#                       #action_probs = action_probs *state[self.a_dim*3:self.a_dim*4] works now!!!
#          05.06.2020   No Joker!!!! Used fixed bug in gameClasses!

# TODO learn more correct steps?!
#       --> Teste ob jede Karte geshifted werden kann (game logic fehler?!!?!)

# Teste für falschen reward gibts -100 und game state wird nicht gesetzt (man kann nochmal spielen!)

# is this a problem of 4 ai Players playing against each other?

# TODO log more relevant DATA to see why.... (breaks after 12 moves)

class Batch(object):
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def __unicode__(self):
        return self.show()
    def __str__(self):
        return self.show()
    def __repr__(self):
        return self.show()

    def show(self):
        return "[{}]{}_{}".format( str(len(self.rewards)), ''.join(str(e) for e in self.rewards) , ''.join(str(e) for e in self.is_terminals))

    def clear_batch(self):
        #print("Clear memory:", len(self.actions), len(self.states), len(self.logprobs), len(self.rewards), len(self.is_terminals))
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def append_batch(self, b):
        self.actions.extend(b.actions)
        self.states.extend(b.states)
        self.logprobs.extend(b.logprobs)
        self.rewards.extend(b.rewards)
        self.is_terminals.extend(b.is_terminals)

    def convert_batch_tensor(self):
        self.states = torch.Tensor(self.states)
        self.actions= torch.Tensor(self.actions)
        self.logprobs = torch.Tensor(self.logprobs)
        #self.states = torch.from_numpy(self.states).float()
        #print(torch.from_numpy(self.states[0]).float())

class Memory:
    def __init__(self):
        self.batches = []
    def appendBatch(self, batch):
        #only append batch if it has data!
        if len(batch.actions)>0:
            self.batches.append(deepcopy(batch))

    def append_memo(self, input_memory):
        self.batches.extend(input_memory.batches)

    def shuffle(self):
        return random.shuffle(self.batches)

    def printMemory(self):
        print("Batches in the Memory:")
        for batch in self.batches:
            print(batch)
    def clear_memory(self):
        del self.batches[:]
    def batch_from_Memory(self):
        # converts to big batch
        b = Batch()
        for i in self.batches:
            b.append_batch(i)
        return b

#Actor Model:
class ActorModel(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorModel, self).__init__()
        self.a_dim   = action_dim

        self.ac      = nn.Linear(state_dim, n_latent_var)
        self.ac_prelu= nn.PReLU()
        self.ac1      = nn.Linear(n_latent_var, n_latent_var)
        self.ac1_prelu= nn.PReLU()

        # Actor layers:
        self.a1      = nn.Linear(n_latent_var+action_dim, action_dim)

        # Critic layers:
        self.c1      = nn.Linear(n_latent_var, n_latent_var)
        self.c1_prelu= nn.PReLU()
        self.c2      = nn.Linear(n_latent_var, 1)

    def forward(self, input):
        # For 4 players each 15 cards on hand:
        # input=on_table(60)+ on_hand(60)+ played(60)+ play_options(60)+ add_states(15)
        # add_states = color free (4)+ would win (1) = 5  for each player
        #input.shape  = 15*4*4=240+3*5 (add_states) = 255

        #Actor and Critic:
        ac = self.ac(input)
        ac = self.ac_prelu(ac)
        ac = self.ac1(ac)
        ac = self.ac1_prelu(ac)

        # Get Actor Result:
        if len(input.shape)==1:
            options = input[self.a_dim*3:self.a_dim*4]
            actor_out =torch.cat( [ac, options], 0)
        else:
            options = input[:, self.a_dim*3:self.a_dim*4]
            actor_out   = torch.cat( [ac, options], 1)
        actor_out = self.a1(actor_out)
        actor_out = actor_out.softmax(dim=-1)

        # Get Critic Result:
        critic = self.c1(ac)
        critic = self.c1_prelu(critic)
        critic = self.c2(critic)

        return actor_out, critic

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()
        self.a_dim   = action_dim

        # actor critic
        self.actor_critic = ActorModel(state_dim, action_dim, n_latent_var)

    def act(self, state, memory):
        if type(state) is np.ndarray:
            state = torch.from_numpy(state).float()
        action_probs, _ = self.actor_critic(state)
        # here make a filter for only possible actions!
        #action_probs = action_probs *state[self.a_dim*3:self.a_dim*4]
        dist = Categorical(action_probs)
        action = dist.sample()# -> gets the lowest non 0 value?!

        if memory is not None:
            #necessary to convet all to numpy otherwise deepcopy not possible!
            memory.states.append(state.data.numpy())
            memory.actions.append(int(action.data.numpy()))
            memory.logprobs.append(float(dist.log_prob(action).data.numpy()))

        return action.item()

    def evaluate(self, state, action):
        action_probs, state_value = self.actor_critic(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy    = dist.entropy()
        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, lr_decay=100000):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas, weight_decay=5e-5) # eps=1e-5
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var)
        self.policy_old.load_state_dict(self.policy.state_dict())
        #TO decay learning rate during training:
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_decay, gamma=0.9)
        self.MseLoss = nn.MSELoss() # MSELossFlat # SmoothL1Loss

    def monteCarloRewards(self, memory):
        # Monte Carlo estimate of state rewards:
        # see: https://medium.com/@zsalloum/monte-carlo-in-reinforcement-learning-the-easy-way-564c53010511
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.append(discounted_reward)
        rewards.reverse()
        # Normalizing the rewards:
        rewards = torch.tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        return rewards

    def calculate_total_loss(self, state_values, logprobs, old_logprobs, advantage, rewards, dist_entropy):
        # 1. Calculate how much the policy has changed                # Finding the ratio (pi_theta / pi_theta__old):
        ratios = torch.exp(logprobs - old_logprobs.detach())
        # 2. Calculate Actor loss as minimum of 2 functions
        surr1       = ratios * advantage
        surr2       = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantage
        actor_loss  = -torch.min(surr1, surr2)
        # 3. Critic loss
        crictic_discount = 0.5
        critic_loss =crictic_discount*self.MseLoss(state_values, rewards)
        # 4. Total Loss
        beta       = 0.005 # encourage to explore different policies
        total_loss = critic_loss+actor_loss- beta*dist_entropy
        return total_loss

    def my_update(self, memory):
        # My rewards: (learns the moves!)
        rewards = torch.tensor(memory.rewards)
        rewards = self.monteCarloRewards(memory)

        # convert list to tensor
        old_states   = memory.states.detach()
        old_actions  = memory.actions.detach()
        old_logprobs = memory.logprobs.detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)
            advantages = rewards - state_values.detach()

            #rewards    = rewards.float()
            #advantages = advantages.float()
            loss       =  self.calculate_total_loss(state_values, logprobs, old_logprobs, advantages, rewards, dist_entropy)

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Reduce lr:
        #self.scheduler.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

@ray.remote
def playRandomSteps(policy, env, steps, max_corr):
    # difference here is that it is played until the END!
    finished_ai_reward  = 0
    finished_games      = 0
    total_ai_reward     = 0
    total_correct       = 0
    finished_random     = 0
    for i in range(steps):
        state  = env.resetRandomPlay_Env()
        done   = 0
        tmp    = 0
        corr_moves = 0
        while not done:
            action = policy.act(state, None)
            state, rewards, corr_moves, done = env.stepRandomPlay_Env(action, print__=False)
            if rewards is not None:
                tmp+=rewards[0]
            else:
                rewards = [-100, -100]
        if done and corr_moves == max_corr:
            finished_ai_reward +=rewards[0]
            finished_random    +=rewards[1]
            finished_games     +=1
        total_correct    +=corr_moves
        total_ai_reward  +=rewards[0]
    return finished_ai_reward, finished_games, total_ai_reward, total_correct, finished_random


def test_with_random(policy, env, jjj, max_corr, episodes=5000, print_game=50):
    finished_ai_reward, finished_games, total_ai_reward, total_correct, finished_random = 0.0, 0.0, 0.0, 0.0, 0.0
    nu_remote            = 10
    steps                = int(episodes/nu_remote)

    result = ray.get([playRandomSteps.remote(policy, env, steps, max_corr) for i in range(nu_remote)])
    for i in result:
        finished_ai_reward += i[0]
        finished_games     += i[1]
        total_ai_reward    += i[2]
        total_correct      += i[3]
        finished_random    += i[4]

    #print every nth game:
    if jjj%print_game == 0:
        state           = env.resetRandomPlay_Env(print__=True)
        done            = 0
        while not done:
            action = policy.act(state, None)
            state, ai_reward, corr_moves, done = env.stepRandomPlay_Env(action, print__=True)
        if done:
            print("Final ai reward:", ai_reward, "moves", corr_moves, "done", done)

    finished_games = int(finished_games)
    if finished_games>0:
        finished_ai_reward = finished_ai_reward/finished_games
        finished_random    = finished_random/finished_games
    total_ai_reward    = total_ai_reward/episodes
    total_correct      = total_correct/episodes

    return finished_games, finished_ai_reward, total_ai_reward, total_correct, finished_random

def generateData():
    print("todo multiprocessing")

def printState(state, env):
    a = env.action_space.n
    #on_table+ on_hand+ played+ play_options+ add_states
    on_table, on_hand, played, options =state[0:a], state[a:a*2], state[a*2:a*3], state[a*3:a*4]
    for i,j in zip([on_table, on_hand, played, options], ["on_table", "on_hand", "played", "options"]):
         print(j, i, env.my_game.state2Cards(i))

@ray.remote
def playSteps(env, policy, steps, max_corr):
    batches = [Batch(), Batch(), Batch(), Batch()]
    result_memory = Memory()
    done   = 0
    state  = env.reset()
    for i in range(steps):
        player =  env.my_game.active_player
        action = policy.act(state, batches[player])# <- state is appended to memory in act function
        state, rewards, done, _ = env.step(action)
        batches[player].is_terminals.append(done)

        if isinstance(rewards, int):
            batches[player].rewards.append(rewards)
        else:
            for jjj in range(4): # delete last element
                batches[jjj].rewards      = batches[jjj].rewards[:len(batches[jjj].is_terminals)-1] # for total of 4 players
                batches[jjj].is_terminals = batches[jjj].is_terminals[:len(batches[jjj].is_terminals)-1]
            for jjj in range(4): # append last element
                batches[jjj].rewards.append(float(rewards[jjj]))
                batches[jjj].is_terminals.append(True)
        if done:
            for jjj in range(4):
                result_memory.appendBatch(batches[jjj])
                batches[jjj].clear_batch()
        if done and (i+max_corr)>steps:
            break
    return result_memory

def learn_single(ppo, update_timestep, eps_decay, env, increase_batch, max_reward=-2.0):
    memory          = Memory()
    eps_decay       = int(eps_decay/update_timestep)
    timestep        = 0
    log_interval    = 2           # print avg reward in the interval
    jjj             = 0
    wrong           = 0.0 # for logging
    nu_remote       = 10 #less is better, more games are finished! for update_timestep=30k 100 is better than 10 here!
    steps           = int(update_timestep/nu_remote)
    increase_batch  = int(increase_batch/nu_remote)
    i_episode       = 0
    max_corr_moves  = (env.action_space.n/4+env.options_test["nu_shift_cards"])
    curr_batch_len  = 0
    print("Batch size:",steps, increase_batch, curr_batch_len)
    for uuu in range(0, 500000000+1):
        #### get Data in parallel:
        ttmp    = datetime.datetime.now()
        result = ray.get([playSteps.remote(env, ppo.policy_old, steps, max_corr_moves) for i in range(nu_remote)])
        for i in result:
            memory.append_memo(i)

        playing_time = round((datetime.datetime.now()-ttmp).total_seconds(),2)
        i_episode    += steps*nu_remote

        # update if its time
        if uuu % 1 == 0:
            #CAUTION MEMORY SIZE INCREASES SLOWLY HERE (during leraning correct moves...)
            # -> TODO use trainloader here! (random minibatch with fixed size)
            # TODO DO NOT SHUFFLE IN THIS WAY! Dann sind alle sequenzen durcheinander!
            memory.shuffle()
            bbb = memory.batch_from_Memory()
            curr_batch_len += len(bbb.actions)
            bbb.convert_batch_tensor()
            if curr_batch_len>0:
                ppo.my_update(bbb)
            del bbb
            memory.clear_memory()
            steps    += increase_batch

        # logging
        if uuu % eps_decay == 0:
            print("do not use eps_decay....")
            #ppo.eps_clip *=0.8

        if uuu % log_interval == 0:
            jjj +=1
            finished_games, finished_ai_reward, total_ai_reward, total_correct, finished_random =  test_with_random(ppo.policy_old, env, jjj, max_corr_moves)
            #test play against random
            aaa = ('Game ,{:07d}, mean_rew of {} finished g. ,{:0.5}, of random ,{:0.5}, corr_moves[max:{:2}] ,{:4.4}, mean_rew ,{:1.3},   {},playing t={}, lr={}, batch={}\n'.format(i_episode, finished_games, float(finished_ai_reward), float(finished_random), int(max_corr_moves), float(total_correct), float(total_ai_reward), datetime.datetime.now()-start_time, playing_time, ppo.scheduler.get_lr(), curr_batch_len))
            print(aaa)
            curr_batch_len = 0
            #max correct moves: 61
            if total_ai_reward>max_reward and total_correct>2.0 and finished_games>10:
                 path =  'PPO_noLSTM_{}_{}_{}_{}'.format(i_episode, finished_ai_reward, finished_games, total_ai_reward)
                 torch.save(ppo.policy.state_dict(), path+".pth")
                 max_reward = total_ai_reward
                 torch.onnx.export(ppo.policy_old.actor_critic, torch.rand(env.observation_space.n), path+".onnx")
                 print("ONNX 1000 Games RESULT:")
                 #testOnnxModel(path+".onnx")

            with open(log_path, "a") as myfile:
                myfile.write(aaa)

def testNotLearned():
    env = gym.make("Witches_multi-v2")
    state_dim  = env.observation_space.n
    action_dim = env.action_space.n
    for i in range(4):
        print(env.my_game.players[i].hand)

    ppo = PPO(state_dim, action_dim, 128, 0.01, (0.9, 0.999), 0.995, 16, 0.2, 10.0)
    ppo.policy.load_state_dict(torch.load("PPO_noLSTM_91777800_0.3446543408360129_4976_-0.137.pth"))  ## PPO_noLSTM_91777800_0.3446543408360129_4976_-0.137 --  PPO_noLSTM_14480000_-0.7211500590783773_2539_-49.5862
    ppo.policy.actor_critic.eval()
    batches = [Batch(), Batch(), Batch(), Batch()]
    finished_random     = 0
    for i in range(1):
        state  = env.resetRandomPlay_Env(print__ = True)
        done   = 0
        tmp    = 0
        corr_moves = 0
        while not done:
            action = ppo.policy_old.act(state, None)
            state, rewards, corr_moves, done = env.stepRandomPlay_Env(action, print__=True)
            print(rewards, done)
            if rewards is not None:
                tmp+=rewards[0]
            else:
                rewards = [-100, -100]


    # result_memory = Memory()
    # done   = 0
    # state  = env.reset()
    # for i in range(1000):
    #     player =  env.my_game.active_player
    #     action = ppo.policy_old.act(state, batches[player])# <- state is appended to memory in act function
    #     state, rewards, done, _ = env.step(action)
    #     print(rewards, done)
    #     batches[player].is_terminals.append(done)
    #
    #     if isinstance(rewards, int):
    #         batches[player].rewards.append(rewards)
    #     else:
    #         for jjj in range(4): # delete last element
    #             batches[jjj].rewards      = batches[jjj].rewards[:len(batches[jjj].is_terminals)-1] # for total of 4 players
    #             batches[jjj].is_terminals = batches[jjj].is_terminals[:len(batches[jjj].is_terminals)-1]
    #         for jjj in range(4): # append last element
    #             batches[jjj].rewards.append(float(rewards[jjj]))
    #             batches[jjj].is_terminals.append(True)
    #     if done:
    #         print(i, done)
    #         print(aaaa)
    #         for jjj in range(4):
    #             result_memory.appendBatch(batches[jjj])
    #             batches[jjj].clear_batch()
    #     if done and (i+max_corr)>steps:
    #         break

    # result = ray.get([playSteps.remote(env, ppo.policy_old, 1000, 10) for i in range(1)])
    # print("\n\n\n", "BATCHES LEN")
    # print(result[0].batches)
    # print(eee)
    # final_memory = Memory()
    # for i in result:
    #     if len(i.batches) > 0: final_memory.append_memo(i)
    # final_memory.shuffle()
    # final_memory.printMemory()

    # bbb = final_memory.batch_from_Memory()
    # bbb.convert_batch_tensor()
    # ppo.my_update(bbb)
    # bbb.clear_batch()

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    ## Setup Env:
    train_path    ="PPO_noLSTM_91777800_0.3446543408360129_4976_-0.137.pth"    ## PPO_noLSTM_91777800_0.3446543408360129_4976_-0.137 --  PPO_noLSTM_14480000_-0.7211500590783773_2539_-49.5862
    #train_path  ="ppo_models/PPO_multi_0.0_36.7445.pth"
    env_name      = "Witches_multi-v2"
    try:
        os.remove(os.getcwd()+"/"+log_path)
    except:
        print("No Logging to be removed!")
    # creating environment
    print("Creating model:", env_name)
    env = gym.make(env_name)

    # Setup General Params
    state_dim  = env.observation_space.n
    action_dim = env.action_space.n

    nu_latent       = 128
    gamma           = 0.99
    K               = 16#5
    update_timestep = 300000 #train further1: 80000  train further2: 180000  train further2: 300000
    increase_batch  = 100 # value is multipled with 10=nu_remote!! increase batch size every update step! (instead of lowering lr)
    log_path      = "logging"+str(K)+"_"+str(update_timestep)+"_"+str(increase_batch)+"_"+str(random.randrange(100))+".txt"

    train_from_start= False

    if train_from_start:
        print("train from start")
        eps = 0.2 # eps=0.2
        lr  = 0.01
        eps_decay       = 8000000
        lr_decay        = eps_decay/update_timestep
        print("Parameters for training:\n", state_dim, action_dim, nu_latent, lr, (0.9, 0.999), gamma, K, eps, lr_decay)
        ppo1 = PPO(state_dim, action_dim, nu_latent, lr, (0.9, 0.999), gamma, K, eps, lr_decay)
        learn_single(ppo1, update_timestep, eps_decay, env, increase_batch)
    else:
        # setup learn further:
        eps_further = 0.001  #train further1: 0.05  train further2: 0.01 train further3:  0.001
        lr_further  = 0.0005  #train further1: 0.001  train further2: 0.0009, train further3 0.0001
        eps_decay   = 8000000
        lr_decay    = 20000
        models = []
        ppo = PPO(state_dim, action_dim, nu_latent, lr_further, (0.9, 0.999), gamma, K, eps_further, lr_decay)
        ppo.policy.load_state_dict(torch.load(train_path))
        ppo.policy.actor_critic.eval()
        learn_single(ppo, update_timestep, eps_decay, env, increase_batch)
