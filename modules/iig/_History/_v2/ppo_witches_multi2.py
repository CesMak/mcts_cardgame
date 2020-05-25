import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import gym_witches_multiv2
import datetime

#Code reference:
#https://github.com/nikhilbarhate99/PPO-PyTorch/blob/master/PPO.py

# For exporting the model:
import torch.onnx
import onnx
import onnxruntime

import numpy as np
import os
import random

class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        #print("Clear memory:", len(self.actions), len(self.states), len(self.logprobs), len(self.rewards), len(self.is_terminals))
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]

    def shuffle(self):
        mapIndexPosition = list(zip(self.actions, self.states, self.logprobs, self.rewards, self.is_terminals))
        random.shuffle(mapIndexPosition)
        self.actions, self.states, self.logprobs, self.rewards, self.is_terminals = map(list, zip(*mapIndexPosition))

#Actor Model:
class ActorMod(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorMod, self).__init__()
        self.a_dim   = action_dim
        self.l1      = nn.Linear(state_dim, n_latent_var)
        self.l1_tanh = nn.PReLU()
        self.l2      = nn.Linear(n_latent_var, n_latent_var)
        self.l2_tanh = nn.PReLU()
        self.l3      = nn.Linear(n_latent_var+action_dim, action_dim)

    def forward(self, input):
        x = self.l1(input)
        x = self.l1_tanh(x)
        x = self.l2(x)
        out1 = self.l2_tanh(x) # 64x1
        if len(input.shape)==1:
            out2 = input[self.a_dim*3:self.a_dim*4]   # 60x1 this are the available options of the active player!
            output =torch.cat( [out1, out2], 0)
        else:
            out2 = input[:, self.a_dim*3:self.a_dim*4]
            output =torch.cat( [out1, out2], 1) #how to do that?
        x = self.l3(output)
        return x.softmax(dim=-1)


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        #TODO see question: https://discuss.pytorch.org/t/pytorch-multiple-inputs-in-sequential/74040
        self.action_layer = ActorMod(state_dim, action_dim, n_latent_var)

        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.PReLU(),#prelu
                nn.Linear(n_latent_var, n_latent_var),
                nn.PReLU(),
                nn.Linear(n_latent_var, 1)
                )

    def forward(self, state_input):
        he = self.act(state_input, None)
        returned_tensor = torch.zeros(1, 2)
        returned_tensor[:, 0] = he#.item()
        return returned_tensor

    def act(self, state, memory):
        if type(state) is np.ndarray:
            state = torch.from_numpy(state).float()
        action_probs = self.action_layer(state)
        # here make a filter for only possible actions!
        #action_probs = action_probs *state[120:180]
        dist = Categorical(action_probs)
        action = dist.sample()

        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(dist.log_prob(action))

        return action.item()

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

class PPO:
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip, lr_decay=1000000):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas, eps=1e-8) # no eps before!
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
            #rewards.insert(0, discounted_reward)
            rewards.append(discounted_reward)
        rewards.reverse()
        # Normalizing the rewards:
        rewards = torch.tensor(rewards)         # use here memory.rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)  # commented out
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
        beta       = 0.01 # encourage to explore different policies TOBI HAT DA 0.005
        total_loss = critic_loss+actor_loss- beta*dist_entropy
        return total_loss

    def my_update(self, memory):
        # My rewards: (learns the moves!)
        rewards = torch.tensor(memory.rewards)
        #rewards = rewards/100
        rewards = self.monteCarloRewards(memory)

        # convert list to tensor
        old_states   = torch.stack(memory.states).detach()
        old_actions  = torch.stack(memory.actions).detach()
        old_logprobs = torch.stack(memory.logprobs).detach()

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

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

def test_with_random(ppo_test, env, jjj, episodes=5000):
    # make this parallel!!!
    # I have 12 kernels?
    total_correct = 0
    total_ai_rewards    = 0
    only_correct_rewards= 0  # without illegal moves, per step reward
    relevant_episodes   = 0
    finished_games      = 0
    finished_corr_rewards= 0
    per_step_reward     = 0
    total_ai_rewardd     = 0
    max_corr_moves       = (env.action_space.n/4+env.options_test["nu_shift_cards"])
    for i in range(episodes):
        state  = env.resetRandomPlay_Env()
        done   = 0
        tmp    = 0
        while not done:
            action = ppo_test.policy_old.act(state, None)
            state, ai_reward, corr_moves, done = env.stepRandomPlay_Env(action, print__=False)
            if ai_reward is not None:
                tmp+=ai_reward
            else:
                ai_reward = -1000
            total_ai_rewards +=ai_reward
        if done and corr_moves == max_corr_moves and finished_games<episodes*0.9:
            total_ai_rewardd+=ai_reward
            finished_games  +=1
        total_correct +=corr_moves
    ##Uncomment for no printing:
    #play a game with printing:
    if jjj%10 == 0:
        for i in range(1):
            state           = env.resetRandomPlay_Env(print__=True)
            done            = 0
            while not done:
                action = ppo_test.policy_old.act(state, None)
                state, ai_reward, corr_moves, done = env.stepRandomPlay_Env(action, print__=True)
            if done:
                print("Final ai reward:", ai_reward, corr_moves, done)
            #print("Total AI Reward in one example game:", total_ai_reward, "corr_moves:", corr_moves, "GameOver", done)
            #print("Episodes:", relevant_episodes, "Rewards:", only_correct_rewards, "Games finished:", finished_games)
    ##Uncomment for no printing^^^^ ^^
    if total_correct==0:
        res1 = 0.0
    else:
        res1 = total_correct/episodes
    if total_ai_rewardd==0.0:
        res2 = 0.0
    else:
        res2 = total_ai_rewardd/finished_games
    return res1, res2, finished_games, max_corr_moves

def generateData():
    print("todo multiprocessing")

def printState(state, env):
    a = env.action_space.n
    on_table, on_hand, played, options =state[0:a], state[a:a*2], state[a*2:a*3], state[a*3:a*4]
    for i,j in zip([on_table, on_hand, played, options], ["on_table", "on_hand", "played", "options"]):
         print(j, i, env.my_game.state2Cards(i))

def learn_single(ppo, update_timestep, eps_decay, env, max_reward=-7.4):
    memory          = Memory()
    total_games_won = np.zeros(4,)
    timestep        = 0
    total_rewards   = 0
    invalid_moves   = 0
    log_interval    = update_timestep*2           # print avg reward in the interval
    total_correct   = 0
    correct_moves   = 0
    jjj             = 0
    wrong           = 0 # for logging
    done            = 0
    state = env.reset()
    for i_episode in range(1, 500000000+1):
        timestep += 1
        #if done: state=env.reset()# moved inside witches_env

        #while not done:
        action = ppo.policy_old.act(state, memory)# -< state is appended to memory in act function
        state, rewards, done, info = env.step(action)
        memory.is_terminals.append(done)

        # if list -> game is finished!
        if isinstance(rewards, int):
            memory.rewards.append(rewards)
        else:
            memory.rewards = memory.rewards[:len(memory.rewards)-3] # for total of 4 players
            memory.rewards.extend(torch.from_numpy(rewards).float())

        total_correct +=info["correct_moves"]

        # update if its time
        if timestep % update_timestep == 0:
            #CAUTION MEMORY SIZE INCREASES SLOWLY HERE (during leraning correct moves...)
            # -> TODO use trainloader here! (random minibatch with fixed size)
            memory.shuffle()
            ppo.my_update(memory)
            memory.clear_memory()
            timestep = 0

        # logging
        if i_episode % eps_decay == 0:
            for lll in range(4):
                ppo.eps_clip *=0.8

        if i_episode % log_interval == 0:
            jjj +=1
            total_correct = total_correct/log_interval
            corr_moves, mean_reward, finished_games, max_corr_moves =  test_with_random(ppo, env, jjj)
            if finished_games>10:
                wrong = (finished_games) *(int(max_corr_moves)-float(corr_moves))
            #test play against random
            aaa = ('Game ,{:07d}, reward per game in {} g. ,{:0.5}, corr_moves[max:{:2}] ,{:4.4}, wrong ,{} ,  Time ,{},\n'.format(i_episode, finished_games, float(mean_reward), int(max_corr_moves), float(corr_moves), int(wrong), datetime.datetime.now()-start_time))
            print(aaa)
            #max correct moves: 61
            if mean_reward>max_reward and corr_moves>2.0 and finished_games>10:
                 path =  'PPO_{}_{}_{}'.format(i_episode, finished_games, mean_reward)
                 torch.save(ppo.policy.state_dict(), path+".pth")
                 max_reward = mean_reward
                 torch.onnx.export(ppo.policy_old.action_layer, torch.rand(env.observation_space.n), path+".onnx")
                 print("ONNX 1000 Games RESULT:")
                 #testOnnxModel(path+".onnx")
                 print("\n\n\n")

            total_correct = 0
            with open(log_path, "a") as myfile:
                myfile.write(aaa)

def learn_multi(ppo, update_timestep, eps_decay, env, max_reward=-2):
    memory = [Memory(), Memory(), Memory(), Memory()]
    total_games_won = np.zeros(4,)
    timestep        = 0
    total_rewards   = 0
    invalid_moves   = 0
    log_interval    = update_timestep*2           # print avg reward in the interval
    total_correct=0
    correct_moves = 0
    jjj           = 0
    for i_episode in range(1, 500000000+1):
        timestep += 1
        state = env.reset()
        done  = 0
        while not done:
            i      = env.my_game.active_player
            #print("\n\nActive player:",i)
            #printState(state, env)
            action = ppo[i].policy_old.act(state, memory[i])
            #print("Action:", action)
            state, rewards, done, info = env.step(action)
            #print("Rewards:", rewards, "done", done, "info", info)
            #print("STATE AFTER")
            #printState(state, env)
            if rewards["ai_reward"] is None: # illegal move
                memory[i].rewards.append(-100)
            else:#shift round ->0 or leagal play move
                memory[i].rewards.append(0)
            memory[i].is_terminals.append(done)
            #print("Rewards Memo  :", memory[i].rewards)
            #print("Terminals Memo:", memory[i].is_terminals)
            if info["round_finished"] and rewards["state"] == "play" and int(rewards["ai_reward"]) != 0:
                win_player = rewards["player_win_idx"]
                del memory[win_player].rewards[-1] # delete last element of list
                memory[win_player].rewards.append(int(rewards["ai_reward"]))
                #print("\nWINPLAYER\n",win_player)
                #print(memory[win_player].rewards)

        total_correct +=info["correct_moves"]

        # update if its time
        if timestep % update_timestep == 0:
            # update with all information!
            for lll in range(4):
                #for j in range(4):
                ppo[lll].my_update(memory[lll])
            for u in range(4):
                memory[u].clear_memory()

        # Nope! Weight sharing is difficult cause it might learn differently!
        # if timestep % (update_timestep*10) == 0:
        #     print("share weights now:")
        #     final_dict = {}
        #     for [dict1, dict2, dict3, dict4] in zip(ppo[0].policy.state_dict().items(), ppo[1].policy.state_dict().items(), ppo[2].policy.state_dict().items(), ppo[3].policy.state_dict().items()):
        #         val1, val2, val3, val4 = dict1[1], dict2[1], dict3[1], dict4[1]
        #         final_dict[dict1[0]] = (val1+val2+val3+val4)/4
        #     for i in range(4):
        #         ppo[i].policy.load_state_dict(final_dict)

        # logging
        if i_episode % eps_decay == 0:
            for lll in range(4):
                ppo[lll].eps_clip *=0.8

        if i_episode % log_interval == 0:
            jjj +=1
            total_correct = total_correct/log_interval
            corr_moves, mean_reward, finished_games =  test_with_random(ppo[0], env, jjj)
            #test play against random
            aaa = ('Game ,{:07d}, reward per game in {} g. ,{:0.5}, corr_moves ,{:4.6},  Time ,{},\n'.format(i_episode, finished_games, float(mean_reward), float(corr_moves), datetime.datetime.now()-start_time))
            print(aaa)
            #max correct moves: 61
            if mean_reward>max_reward and corr_moves>2.0:
                 path =  'PPO_{}_{}_{}'.format(i_episode, finished_games, mean_reward)
                 torch.save(ppo[0].policy.state_dict(), path+".pth")
                 max_reward = mean_reward
                 torch.onnx.export(ppo[0].policy_old.action_layer, torch.rand(env.observation_space.n), path+".onnx")
                 print("ONNX 1000 Games RESULT:")
                 #testOnnxModel(path+".onnx")
                 print("\n\n\n")

            total_correct = 0
            with open(log_path, "a") as myfile:
                myfile.write(aaa)

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    ## Setup Env:
    train_path    ="PPO_520000_4500_-1.1546666666666667.pth"
    #train_path  ="ppo_models/PPO_multi_0.0_36.7445.pth"
    env_name      = "Witches_multi-v2"
    log_path      = "logging.txt"
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

    nu_latent       = 128#64
    gamma           = 0.99
    K               = 16#5
    update_timestep = 30000 #2000 # 50000 memory error..... -> mit 35000 nicht so gut (nach 6,5h) wie mit 20000 nach 5h, 20k for 17 moves memory error!


    train_from_start= True
    train_single    = True

    if train_from_start:
        print("train from start")
        eps = 0.2 # eps=0.2
        lr  = 0.001
        eps_decay       = 200000
        lr_decay        = 200000
        ppo1 = PPO(state_dim, action_dim, nu_latent, lr, (0.9, 0.999), gamma, K, eps, lr_decay)
        if train_single:
            learn_single(ppo1, update_timestep, eps_decay, env)
        else:
            ppo2 = PPO(state_dim, action_dim, nu_latent, lr, (0.9, 0.999), gamma, K, eps, lr_decay)
            ppo3 = PPO(state_dim, action_dim, nu_latent, lr, (0.9, 0.999), gamma, K, eps, lr_decay)
            ppo4 = PPO(state_dim, action_dim, nu_latent, lr, (0.9, 0.999), gamma, K, eps, lr_decay)
            learn_multi([ppo1, ppo2, ppo3, ppo4], update_timestep, eps_decay, env)
    else:
        # setup learn further:
        eps_further = 0.1
        lr_further  = 0.0005
        eps_decay   = 20000
        lr_decay    = 20000
        models = []
        if train_single:
            ppo = PPO(state_dim, action_dim, nu_latent, lr_further, (0.9, 0.999), gamma, K, eps_further, lr_decay)
            ppo.policy.load_state_dict(torch.load(train_path))
            ppo.policy.action_layer.eval()
            ppo.policy.value_layer.eval()
            learn_single(ppo, update_timestep, eps_decay, env)
        else:
            for i in range(4):
                ppo = PPO(state_dim, action_dim, nu_latent, lr_further, (0.9, 0.999), gamma, K, eps_further, lr_decay)
                ppo.policy.load_state_dict(torch.load(train_path))
                ppo.policy.action_layer.eval()
                ppo.policy.value_layer.eval()
                models.append(ppo)

            learn_multi(models, update_timestep, eps_decay, env)
