#https://github.com/seungeunrho/minimalRL/blob/master/ppo.py
import os
import gym
import gym_witches
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

#Max Reward to achieve: 8.5+1
#result_reward+=(rewards[self.reinfo_index]+21)/26  -->8.5+17*(21/26)=8.5+13.73=22.23 (für nuller runde!)
# 13.73 = nuller runde!
#Max corr moves: 17

#Hyperparameters (for starting)
learning_rate = 0.0005
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.1
K_epoch       = 3
T_horizon     = 20  #update_timestep
# beta
# eps_clip adam

# for learning further:
# learning_rate = 0.0000005
# gamma         = 0.98
# lmbda         = 0.99   #0.9 to 1
# eps_clip      = 0.005
# K_epoch       = 3
# T_horizon     = 20  #update_timestep

#Model Parameter
latent_layers  = 256 #64 was not as good as 256

#Logging params
print_interval = 2000


log_path       = "logging.txt"
print(os.getcwd()+"/"+log_path)
try:
    os.remove(os.getcwd()+"/"+log_path)
except:
    print("No Logging to be removed!")
start_time = datetime.datetime.now()

env   = gym.make("Witches-v0")

class ActorMod(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorMod, self).__init__()
        self.l1      = nn.Linear(state_dim, n_latent_var)
        self.l1_tanh = nn.ReLU()
        self.l2      = nn.Linear(n_latent_var, n_latent_var)
        self.l2_tanh = nn.ReLU()
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

class PPO(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(PPO, self).__init__()
        self.data = []

        #value function: fc1 layer
        #viel besser als 3 layer von zuvor!
        self.fc1   = nn.Linear(state_dim,n_latent_var)
        self.fc_v  = nn.Linear(n_latent_var,1)

        #action:layer:
        self.action_layer = ActorMod(state_dim, action_dim, n_latent_var)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        #self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=100000, gamma=0.9)
    def v(self, x):
        x = F.relu(self.fc1(x))
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, done_lst = [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s,a,r,s_prime,done_mask, prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                          torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                          torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s, a, r, s_prime, done_mask, prob_a

    def train_net(self):
        s, a, r, s_prime, done_mask, prob_a = self.make_batch()

        for i in range(K_epoch):
            td_target = r + gamma * self.v(s_prime) * done_mask
            delta = td_target - self.v(s)
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for delta_t in delta[::-1]:
                advantage = gamma * lmbda * advantage + delta_t[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi = self.action_layer(s)
            pi_a = pi.gather(1,a)
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == exp(log(a)-log(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(self.v(s) , td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        #self.scheduler.step()#sheduler to lower the learning rate !

def train(model):
    timestep = 0
    score = 0.0
    moves = 0.0 # correct moves.
    games_won = np.zeros(4,)
    total_ai  = 0.0
    save_model_min =0.0
    for n_epi in range(100000000):
        timestep += 1
        s = env.reset()
        done = False
        while not done:
            prob = model.action_layer(torch.from_numpy(s).float())
            m = Categorical(prob)
            a = m.sample().item()
            state, r, done, info = env.step(a)
            model.put_data((s, a, r, state, prob[a].item(), done))
            s = state
        score    += r
        moves    +=info["correct_moves"]
        games_won+=info["number_of_won"]
        total_ai +=info["total"]

        if timestep % T_horizon == 0:
            model.train_net()
            timestep = 0

        if n_epi%print_interval==0 and n_epi!=0:
            mean_score = score/print_interval
            mean_moves = moves/print_interval
            mean_total = total_ai/print_interval
            aaa = ('Game ,{:07d}, reward ,{:0.5}, corr ,{:4.4}, games_won ,{}, total ,{},  Time ,{},\n'.format(n_epi, mean_score, mean_moves, games_won,mean_total, datetime.datetime.now()-start_time))
            print(aaa)
            if (mean_score>=save_model_min):
                save_model_min = mean_score
                path =  'ppo_models/PPO_{}_{}_{}'.format("Witches-v0", mean_score, games_won[1])
                torch.save(model.action_layer.state_dict(), path+".pth")
                torch.onnx.export(model.action_layer, torch.rand(240), path+".onnx")
            with open(log_path, "a") as myfile:
                myfile.write(aaa)
            score      = 0.0
            moves      = 0.0
            total_ai   = 0.0
            games_won  = np.zeros(4,)


def learn_further(pretrained_path):
    model = PPO(240, 60, latent_layers)
    model.action_layer.load_state_dict(torch.load(pretrained_path))
    train(model)

def main():
    model = PPO(240, 60, latent_layers)
    train(model)

if __name__ == '__main__':
    main()
    #learn_further("ppo_models/PPO_Witches-v0_11.305249999992009_101.0.pth")
