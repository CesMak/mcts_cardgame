import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import gym_witches_multi
import datetime

# For exporting the model:
import torch.onnx
import onnx
import onnxruntime

import numpy as np
import os

#used for testing:
from gameClasses import player

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

class ActorMod(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorMod, self).__init__()
        self.l1      = nn.Linear(state_dim, n_latent_var)
        self.l1_tanh = nn.PReLU()
        self.l2      = nn.Linear(n_latent_var, n_latent_var)
        self.l2_tanh = nn.PReLU()
        self.l3      = nn.Linear(n_latent_var+60, action_dim)

    def forward(self, input):
        x = self.l1(input)
        x = self.l1_tanh(x)
        x = self.l2(x)
        out1 = self.l2_tanh(x) # 64x1
        if len(input.shape)==1:
            out2 = input[180:240]   # 60x1 this are the available options of the active player!
            output =torch.cat( [out1, out2], 0)
        else:
            out2 = input[:, 180:240]
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
        #what values are returned here?
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
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas, eps=1e-5) # no eps before!
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var)
        self.policy_old.load_state_dict(self.policy.state_dict())
        #TO decay learning rate during training:
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_decay, gamma=0.9)
        self.MseLoss = nn.MSELoss()

    def monteCarloRewards(self, memory):
        # Monte Carlo estimate of state rewards:
        # see: https://medium.com/@zsalloum/monte-carlo-in-reinforcement-learning-the-easy-way-564c53010511
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(rewards)         # use here memory.rewards
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)  # commented out
        return rewards

    def test_rewards(self, rewards, gamma=0.99):
        returns = []
        #returns[-1] = next_value
        for step in reversed(range(rewards)):
            print(step)
            #returns[step] = returns[step + 1] * \ gamma + returns[step]

    def calculate_total_loss(self, state_values, logprobs, old_logprobs, advantage, rewards, dist_entropy):
        # 1. Calculate how much the policy has changed
        ratios = torch.exp(logprobs - old_logprobs.detach())
        # 2. Calculate Actor loss as minimum of 2 functions
        surr1       = ratios * advantage
        surr2       = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantage
        actor_loss  = -torch.min(surr1, surr2)
        # 3. Critic loss
        crictic_discount = 0.5
        critic_loss =crictic_discount*self.MseLoss(state_values, torch.tensor(rewards))
        # 4. Total Loss
        beta       = 0.01 # encourage to explore different policies
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
            loss       =  self.calculate_total_loss(state_values, logprobs, old_logprobs, advantages, rewards, dist_entropy)

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

def state2Cards(in_state):
    player_test    =player("test_name")
    result = [] # on_table, on_hand, played, play_options
    for i in [ in_state[0:60], in_state[60:120], in_state[120:180], in_state[180:240]]:
        result.append( player_test.convertAllCardState(i))
    for j,k in zip(result,["on_table", "on_hand", "played", "options"]):
        print("\t", k, len(j), j, "\n")

def test_trained_model(ppo_test, env_test):
    #print("\ninside test_trained_Model")
    episodes                = 10
    total_results           = np.zeros(4,)
    nu_gameOverReached      = 0
    finished_rounds         = 0
    per_step_reward         = 0
    for i in range(episodes):
        done  = 0
        state = env_test.reset()
        while not done:
            action           = ppo_test.policy_old.act(state, None)
            _, _, done, info = env_test.step(action)
            total_results    += np.asarray(info["rewards"])
            if info["gameOver"]:
                nu_gameOverReached+=1
            if info["round_finished"]:
                finished_rounds   +=1
    return total_results/episodes, finished_rounds/episodes, nu_gameOverReached/episodes

def test_with_random(ppo_test, env, jjj, episodes=50):
    total_correct_moves = 0
    total_ai_rewards    = 0
    only_correct_rewards= 0  # without illegal moves, per step reward
    relevant_episodes   = 0
    finished_games      = 0
    finished_corr_rewards= 0
    per_step_reward     = 0
    for i in range(episodes):
        state  = env.resetRandomPlay_Env()
        done   = 0
        tmp    = 0
        while not done:
            action = ppo_test.policy_old.act(state, None)
            state, ai_reward, corr_moves, done = env.stepRandomPlay_Env(action)
            if ai_reward is not None:
                tmp+=ai_reward
            else:
                ai_reward = -100
            total_ai_rewards +=ai_reward
        if corr_moves>2:
            only_correct_rewards+=tmp/(corr_moves-2) # cause 15 play moves only!
            relevant_episodes +=1
        if corr_moves==17:
            finished_corr_rewards+=tmp/(corr_moves-2)
            finished_games+=1
        total_correct_moves +=corr_moves
    # if relevant_episodes>0:
    #     per_step_reward = only_correct_rewards/relevant_episodes
    if finished_games>0:
        per_step_reward  =finished_corr_rewards/finished_games

    ##Uncomment for no printing:
    #play a game with printing:
    if jjj%10 == 0:
        for i in range(1):
            state           = env.resetRandomPlay_Env()
            done            = 0
            total_ai_reward = 0
            while not done:
                action = ppo_test.policy_old.act(state, None)
                state, ai_reward, corr_moves, done = env.stepRandomPlay_Env(action, print=True)
                if ai_reward is not None:
                    total_ai_reward+=ai_reward
            print("Total AI Reward in one example game:", total_ai_reward, "corr_moves:", corr_moves, "GameOver", done)
            print("Episodes:", relevant_episodes, "Rewards:", only_correct_rewards, "Games finished:", finished_games)
    ##Uncomment for no printing^^^^ ^^
    return total_correct_moves/episodes, per_step_reward

def learn_multi(ppo, update_timestep, eps_decay, env, max_reward=10):
    memory = [Memory(), Memory(), Memory(), Memory()]
    total_games_won = np.zeros(4,)
    timestep        = 0
    total_rewards   = 0
    total_number_of_games_played = 0
    invalid_moves   = 0
    log_interval    = update_timestep           # print avg reward in the interval
    total_correct_moves=0
    correct_moves = 0
    jjj           = 0
    for i_episode in range(1, 500000000+1):
        timestep += 1
        state = env.reset()
        done  = 0
        while not done:
            i      = env.my_game.active_player
            #print("\n\nActive player:",i)
            #state2Cards(state)
            action = ppo[i].policy_old.act(state, memory[i])
            #print("Action:", action)
            state, rewards, done, info = env.step(action)
            #print("Rewards:", rewards, "done", done, "info", info)
            #print("STATE AFTER")
            #state2Cards(state)
            if rewards["ai_reward"] is None: # illegal move
                memory[i].rewards.append(-100)
            else:#shift round ->0 or leagal play move
                memory[i].rewards.append(0)
            memory[i].is_terminals.append(done)
            #print(memory[i].rewards)
            #print(memory[i].is_terminals)
            if info["round_finished"] and rewards["state"] == "play" and int(rewards["ai_reward"]) != 0:
                win_player = rewards["player_win_idx"]
                del memory[win_player].rewards[-1] # delete last element of list
                memory[win_player].rewards.append(int(rewards["ai_reward"]))
                #print("\nWINPLAYER\n",win_player)
                #print(memory[win_player].rewards)
                #print(eee)

        total_correct_moves +=info["correct_moves"]

        # update if its time
        if timestep % update_timestep == 0:
            #container = nn.Container()
            for i in range(4):
                ppo[i].my_update(memory[i])
                memory[i].clear_memory()
                #container:add(ppo[i])
            #### weight sharing:    https://discuss.pytorch.org/t/multiplayer-weight-sharing-of-exact-same-network/76282
            #params = container.parameters()
            timestep = 0

        # logging
        if i_episode % eps_decay == 0:
            for lll in range(4):
                ppo[lll].eps_clip *=0.8

        if i_episode % log_interval == 0:
            jjj +=1
            total_correct_moves = total_correct_moves/log_interval
            corr_moves, mean_reward =  test_with_random(ppo[0], env, jjj)
            #test play against random
            aaa = ('Game ,{:07d}, reward ,{:0.5}, test_corr_moves ,{:4.4}, games_won ,{},  corr,{:.2f},Time ,{},\n'.format(i_episode, float(mean_reward), float(corr_moves), "0.0", total_correct_moves, datetime.datetime.now()-start_time))
            print(aaa)
            #max correct moves: 61
            if total_correct_moves>max_reward and corr_moves>5:
                 path =  'ppo_models/PPO_{}_{}_{}'.format("multi", 0.0, total_correct_moves)
                 torch.save(ppo[0].policy.state_dict(), path+".pth")
                 max_reward = total_correct_moves
                 torch.onnx.export(ppo[0].policy_old.action_layer, torch.rand(303), path+".onnx")
                 print("ONNX 1000 Games RESULT:")
                 #testOnnxModel(path+".onnx")
                 print("\n\n\n")

            total_correct_moves = 0
            with open(log_path, "a") as myfile:
                myfile.write(aaa)

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    ## Setup Env:
    train_path  ="ppo_models/PPO_multi_0.0_66.7365.pth"
    #train_path  ="ppo_models/PPO_multi_0.0_36.7445.pth"
    env_name      = "Witches_multi-v1"
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

    nu_latent       = 64
    gamma           = 0.99
    K               = 5
    update_timestep = 2000


    train_from_start= True

    if train_from_start:
        print("train from start")
        eps = 0.1
        lr  = 0.0025
        eps_decay       = 200000
        lr_decay        = 200000
        ppo1 = PPO(state_dim, action_dim, nu_latent, lr, (0.9, 0.999), gamma, K, eps, lr_decay)
        ppo2 = PPO(state_dim, action_dim, nu_latent, lr, (0.9, 0.999), gamma, K, eps, lr_decay)
        ppo3 = PPO(state_dim, action_dim, nu_latent, lr, (0.9, 0.999), gamma, K, eps, lr_decay)
        ppo4 = PPO(state_dim, action_dim, nu_latent, lr, (0.9, 0.999), gamma, K, eps, lr_decay)
        learn_multi([ppo1, ppo2, ppo3, ppo4], update_timestep, eps_decay, env)
    else:
        # setup learn further:
        eps_further = 0.05
        lr_further  = 0.00025
        eps_decay   = 20000
        lr_decay    = 20000
        ppo = PPO(state_dim, action_dim, nu_latent, lr_further, (0.9, 0.999), gamma, K, eps_further, lr_decay)

        ppo.policy.load_state_dict(torch.load(train_path))
        ppo.policy.action_layer.eval()
        ppo.policy.value_layer.eval()
        learn_multi(ppo, update_timestep, eps_decay, env, env_test)
