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
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=lr_decay, gamma=0.5)
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

def test_trained_model(ppo_test, env_test):
    print("\ninside test_trained_Model")
    episodes                = 10
    total_results           = np.zeros(4,)
    nu_gameOverReached      = 0
    finished_rounds         = 0
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


def learn_multi(ppo, update_timestep, eps_decay, env_test):
    memory = Memory()
    total_games_won = np.zeros(4,)
    timestep        = 0
    total_rewards   = 0
    total_number_of_games_played = 0
    invalid_moves   = 0
    log_interval    = update_timestep           # print avg reward in the interval
    max_reward      = -340
    total_correct_moves=0
    correct_moves = 0

    for i_episode in range(1, 500000000+1):
        timestep += 1
        state = env.reset()
        done  = 0
        while not done:
            action = ppo.policy_old.act(state, memory)
            state, rewards, done, info = env.step(action)
            if None in rewards:
                memory.rewards.append(-1000)
            else:
                memory.rewards.append(0.0)
            memory.is_terminals.append(done)
            if info["round_finished"]:
                del memory.rewards[-4:] # delete last 4 elements of list
                memory.rewards.extend(rewards)

        total_correct_moves +=info["correct_moves"]

        # update if its time
        if timestep % update_timestep == 0:
            ppo.my_update(memory)
            memory.clear_memory()
            timestep = 0

        # logging
        if i_episode % eps_decay == 0:
            ppo.eps_clip *=0.4

        if i_episode % log_interval == 0:
            total_correct_moves = total_correct_moves/log_interval
            rewards_mean, _, _ =  test_trained_model(ppo, env_test)
            #test play against random
            aaa = ('Game ,{:07d}, reward ,{:0.5}, invalid_moves ,{:4.4}, games_won ,{},  corr,{:.2f},Time ,{},\n'.format(i_episode, rewards_mean[1], 0.0, "0.0", total_correct_moves, datetime.datetime.now()-start_time))
            print(aaa)
            # if total_correct_moves>0.0:
            #      path =  'ppo_models/PPO_{}_{}_{}'.format("multi", 0.0, total_correct_moves)
            #      torch.save(ppo.policy.state_dict(), path+".pth")
            #      torch.onnx.export(ppo.policy_old.action_layer, torch.rand(303), path+".onnx")
            #      print("ONNX 1000 Games RESULT:")
            #      #testOnnxModel(path+".onnx")
            #      print("\n\n\n")

            total_correct_moves = 0
            with open(log_path, "a") as myfile:
                myfile.write(aaa)

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    ## Setup Env:
    train_path  ="ppo_models/PPO_Witches-v0_21.543865384618478_29587.0.pth"
    env_name      = "Witches_multi-v1"
    log_path      = "logging.txt"
    try:
        os.remove(os.getcwd()+"/"+log_path)
    except:
        print("No Logging to be removed!")
    # creating environment
    print("Creating model:", env_name)
    env = gym.make(env_name)

    # creating testing environment (Play against Random players)
    env_test      = gym.make("Witches_test-v1")

    # Setup General Params
    state_dim  = env.observation_space.n
    action_dim = env.action_space.n

    nu_latent       = 128
    gamma           = 0.999
    K               = 5
    update_timestep = 2000


    train_from_start= True

    if train_from_start:
        print("train from start")
        eps = 0.1
        lr  = 0.0025
        eps_decay       = 20000000
        lr_decay        = 20000000
        ppo = PPO(state_dim, action_dim, nu_latent, lr, (0.9, 0.999), gamma, K, eps, lr_decay)

        learn_multi(ppo, update_timestep, eps_decay, env_test)
    else:
        # setup learn further:
        eps_further = 0.005
        lr_further  = 0.000025
        eps_decay   = 20000
        lr_decay    = 20000
        ppo = PPO(state_dim, action_dim, nu_latent, lr_further, (0.9, 0.999), gamma, K, eps_further, lr_decay)

        ppo.policy.load_state_dict(torch.load(train_path))
        ppo.policy.action_layer.eval()
        ppo.policy.value_layer.eval()
        learn(ppo, update_timestep, eps_decay)
