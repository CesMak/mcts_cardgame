#PPO-LSTM
import gym
import gym_witches_multiv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import time
import numpy as np
import datetime

# For exporting the model:
import torch.onnx
import onnx
import onnxruntime

import numpy as np
import os

#Hyperparameters
learning_rate = 0.001
gamma         = 0.98
lmbda         = 0.95
eps_clip      = 0.4
K_epoch       = 4
T_horizon     = 20

class PPO(nn.Module):
    def __init__(self, action_dim, state_dim):
        super(PPO, self).__init__()
        self.data = []
        self.a_dim= action_dim

        self.fc1   = nn.Linear(state_dim, 64)
        self.lstm  = nn.LSTM(64,    32)
        self.fc_pi = nn.Linear(32+self.a_dim, self.a_dim)
        self.fc_v  = nn.Linear(32,1)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate, eps=1e-5)

    def pi(self, input, hidden):
        #print(len(hidden), hidden[0].shape) #2 torch.Size([1, 1, 32])
        x = F.relu(self.fc1(input))
        x = x.view(-1, 1, 64)
        out1, lstm_hidden = self.lstm(x, hidden)
        batches = 1
        if len(input.shape)>1:
            batches    = input.shape[0]
        out2 = torch.zeros([batches, 1, self.a_dim])
        if len(input.shape) ==1:
            out2[:][0] = input[self.a_dim*3:self.a_dim*4]
        else:
            for i in range(batches):
                out2[i][0] = input[i, self.a_dim*3:self.a_dim*4]
        out  = torch.cat([out1, out2], 2)
        x = self.fc_pi(out)
        prob = F.softmax(x, dim=2)  # shape of x: batchesx1x60
        return prob, lstm_hidden    # shape of prob: batchesx1x60

    def v(self, x, hidden):
        x = F.relu(self.fc1(x))
        x = x.view(-1, 1, 64)
        x, lstm_hidden = self.lstm(x, hidden)
        v = self.fc_v(x)
        return v

    def put_data(self, transition):
        self.data.append(transition)

    def make_batch(self):
        s_lst, a_lst, r_lst, s_prime_lst, prob_a_lst, h_in_lst, h_out_lst, done_lst = [], [], [], [], [], [], [], []
        for transition in self.data:
            s, a, r, s_prime, prob_a, h_in, h_out, done = transition

            s_lst.append(s)
            a_lst.append([a])
            r_lst.append([r])
            s_prime_lst.append(s_prime)
            prob_a_lst.append([prob_a])
            h_in_lst.append(h_in)
            h_out_lst.append(h_out)
            done_mask = 0 if done else 1
            done_lst.append([done_mask])

        s,a,r,s_prime,done_mask,prob_a = torch.tensor(s_lst, dtype=torch.float), torch.tensor(a_lst), \
                                         torch.tensor(r_lst), torch.tensor(s_prime_lst, dtype=torch.float), \
                                         torch.tensor(done_lst, dtype=torch.float), torch.tensor(prob_a_lst)
        self.data = []
        return s,a,r,s_prime, done_mask, prob_a, h_in_lst[0], h_out_lst[0]

    def train_net(self):
        s,a,r,s_prime,done_mask, prob_a, (h1_in, h2_in), (h1_out, h2_out) = self.make_batch()
        first_hidden  = (h1_in.detach(), h2_in.detach())
        second_hidden = (h1_out.detach(), h2_out.detach())

        for i in range(K_epoch):
            v_prime = self.v(s_prime, second_hidden).squeeze(1)
            td_target = r + gamma * v_prime * done_mask
            v_s = self.v(s, first_hidden).squeeze(1)
            delta = td_target - v_s
            delta = delta.detach().numpy()

            advantage_lst = []
            advantage = 0.0
            for item in delta[::-1]:
                advantage = gamma * lmbda * advantage + item[0]
                advantage_lst.append([advantage])
            advantage_lst.reverse()
            advantage = torch.tensor(advantage_lst, dtype=torch.float)

            pi, _ = self.pi(s, first_hidden)  # a.shape = [batches, 1]
            pi_a = pi.squeeze(1).gather(1, a) #pi.squeeze(1).shape = [batches, 60],  pi.squeeze(1).gather(1, a).shape=batchesx1
            ratio = torch.exp(torch.log(pi_a) - torch.log(prob_a))  # a/b == log(exp(a)-exp(b))

            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1-eps_clip, 1+eps_clip) * advantage
            loss = -torch.min(surr1, surr2) + F.smooth_l1_loss(v_s, td_target.detach())

            self.optimizer.zero_grad()
            loss.mean().backward(retain_graph=True)
            self.optimizer.step()

def test_with_random(model, env, jjj, episodes=5000):
    total_correct_moves = 0
    total_ai_rewards    = 0
    only_correct_rewards= 0  # without illegal moves, per step reward
    relevant_episodes   = 0
    finished_games      = 0
    finished_corr_rewards= 0
    per_step_reward     = 0
    total_ai_rewardd     = 0
    for i in range(episodes):
        s      = env.resetRandomPlay_Env()
        h_out  = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
        done   = 0
        tmp    = 0
        while not done:
            h_in = h_out
            prob, h_out = model.pi(torch.from_numpy(s).float(), h_in)
            prob = prob.view(-1)
            m = Categorical(prob)
            a = m.sample().item()
            s_prime, ai_reward, corr_moves, done = env.stepRandomPlay_Env(a)
            if ai_reward is not None:
                tmp+=ai_reward
            else:
                ai_reward = -100
            total_ai_rewards +=ai_reward
            s = s_prime
        if done and corr_moves == env.action_space.n/4 and finished_games<episodes*0.9:
            total_ai_rewardd+=ai_reward
            finished_games  +=1
        total_correct_moves +=corr_moves

    ##Uncomment for no printing:
    #play a game with printing:
    if jjj%10 == 0:
        for i in range(1):
            s  = env.resetRandomPlay_Env()
            h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
            done   = 0
            while not done:
                h_in = h_out
                prob, h_out = model.pi(torch.from_numpy(s).float(), h_in)
                prob = prob.view(-1)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, ai_reward, corr_moves, done = env.stepRandomPlay_Env(a, print=True)
                if done:
                    print("Final ai reward:", ai_reward, corr_moves, done)
            ##Uncomment for no printing^^^^ ^^
    if total_correct_moves==0:
        res1 = 0.0
    else:
        res1 = total_correct_moves/episodes
    if total_ai_rewardd==0.0:
        res2 = 0.0
    else:
        res2 = total_ai_rewardd/finished_games
    return res1, res2, finished_games

def learn_multi(model, update_timestep, env, max_reward=-2):
    log_interval = update_timestep
    total_correct_moves=0
    correct_moves = 0
    my_rewards    = [0, 0, 0, 0] # used for correct reward adding (if round is finished)
    max_reward    = 1
    jjj           = 0
    for i_episode in range(100000000):
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
        s = env.reset()
        done = False
        while not done:
            for t in range(T_horizon):
                i      = env.my_game.active_player
                h_in = h_out
                prob, h_out = model[i].pi(torch.from_numpy(s).float(), h_in)
                prob = prob.view(-1)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)
                if r["ai_reward"] is None: # illegal move
                    rr=-1
                else:#shift round ->0 or leagal play move
                    rr=0
                model[i].put_data((s, a, rr, s_prime, prob[a].item(), h_in, h_out, done))
                if info["round_finished"] and r["state"] == "play" and int(r["ai_reward"]) is not None:
                    for u in range(4):
                        last_transition= model[u].data
                        if len(model[u].data)>1:
                            last_transition= model[u].data[:-1]
                        last_transition_list = list(last_transition[0])
                        last_transition_list[2] =  (int(r["final_rewards"][u])+60)/40
                        last_transition[0] = last_transition_list
                        model[u].data[:-1] = last_transition

                    # win_player = r["player_win_idx"]
                    # last_transition= model[win_player].data
                    # if len(model[win_player].data)>1:
                    #     last_transition= model[win_player].data[:-1]
                    # last_transition_list = list(last_transition[0])
                    # last_transition_list[2] =  int(r["final_rewards"][win_player])
                    # last_transition[0] = last_transition_list
                    # model[win_player].data[:-1] = last_transition

                s = s_prime

                if done:
                    break
            #for lll in range(4):
            model[i].train_net()
        total_correct_moves +=info["correct_moves"]

        if i_episode % log_interval == 0:
            jjj +=1
            total_correct_moves = total_correct_moves/log_interval
            corr_moves, mean_reward, finished_games =  test_with_random(model[0], env, jjj)
            #test play against random
            aaa = ('Game ,{:07d}, reward per game in {} g. ,{:0.5}, corr_moves ,{:4.4},  Time ,{},\n'.format(i_episode, finished_games, float(mean_reward), float(corr_moves), datetime.datetime.now()-start_time))
            print(aaa)
            #max correct moves: 61
            if mean_reward>max_reward and corr_moves>2.0:
                 path =  'PPO_{}_{}_{}'.format(i_episode, finished_games, mean_reward)
                 torch.save(model[0].state_dict(), path+".pth")
                 max_reward = mean_reward
                 print("exported path \n")

            total_correct_moves = 0
            with open(log_path, "a") as myfile:
                myfile.write(aaa)

def main():
    env= gym.make("Witches_multi-v2")
    model = PPO()
    score = 0.0
    print_interval = 2000
    moves          = 0

    for n_epi in range(100000000):
        h_out = (torch.zeros([1, 1, 32], dtype=torch.float), torch.zeros([1, 1, 32], dtype=torch.float))
        s = env.reset()
        done = False

        while not done:
            for t in range(T_horizon):
                h_in = h_out
                prob, h_out = model.pi(torch.from_numpy(s).float(), h_in)
                prob = prob.view(-1)
                m = Categorical(prob)
                a = m.sample().item()
                s_prime, r, done, info = env.step(a)

                model.put_data((s, a, r, s_prime, prob[a].item(), h_in, h_out, done))
                s = s_prime

                score += r
                if done:
                    moves    +=info["correct_moves"]
                    break

            model.train_net()

        if n_epi%print_interval==0 and n_epi!=0:
            print("# of episode :{}, avg score : {:.4f}  {:.4f}".format(n_epi, score/print_interval,  moves/print_interval))
            score = 0.0
            moves = 0.0

    env.close()

if __name__ == '__main__':
    start_time = datetime.datetime.now()
    ## Setup Env:
    train_path    ="ppo_models/PPO_multi_0.0_2.8905.pth"
    env_name      ="Witches_multi-v2"
    log_path      = "logging.txt"
    try:
        os.remove(os.getcwd()+"/"+log_path)
    except:
        print("No Logging to be removed!")
    # creating environment
    print("Creating model:", env_name)
    env = gym.make(env_name)
    state_dim  = env.observation_space.n
    action_dim = env.action_space.n
    update_timestep = 2000


    train_from_start= True

    if train_from_start:
        print("train from start")
        model = []
        for i in range(4):
            model.append(PPO(action_dim, state_dim))
        learn_multi(model, update_timestep, env, max_reward=-2)
    else:
        ppos = [PPO(), PPO(), PPO(), PPO()]
        for i in ppos:
            i.load_state_dict(torch.load(train_path))

        learn_multi(ppos, env)
