import torch
import torch.nn as nn
from torch.distributions import Categorical
import gym
import gym_witches
import stdout
import datetime

# For exporting the model:
import torch.onnx
import onnx
import onnxruntime

import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## # TODO:
# Schaue wie big2 hearts aufgebaut ist was sind die hyperparameter was die rewards? (discount factor?)
# Nute dann das hier!

# 1. Dass keine Invalid moves!
# Feed in Inputs of Hand card in output options!
# see: https://discuss.pytorch.org/t/how-to-concatenate-two-layers-using-sefl-add-module/28922/2
# 2. Schau dass je weiter gespielt wird in einer runde desto höher der Reward! (damit bis ans Ende kommt!)


# 5, 5, 0.0002 funktioniert ganz gut!
# Bestes ergebnis -0.36 als mean reward (pro Zug)
#Learning Parameters: 0.0003 5 64 5 4 0.3 (0.9, 0.999)
#Episode 50 	 reward_mean: -0.92277	-1.13e+02 	 wrong_moves: 870	0:00:31.805750

# other good stats: with esp=1e-05:
# [1.0001e+04 9.0000e+00 3.5580e+03 3.2070e+03]  (9 games wone in <20k games)
# example of invalid moves: Episode 17050 	 reward_mean: -0.045368	-25.5 	 invalid_moves: 2791	0:36:40.401963


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

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, n_latent_var):
        super(ActorCritic, self).__init__()

        # actor
        self.action_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, action_dim),
                # here combine layers!
                nn.Softmax(dim=-1)
                )

        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, n_latent_var),
                nn.Tanh(),
                nn.Linear(n_latent_var, 1)
                )

    def forward(self):
        raise NotImplementedError

    def act(self, state, memory):
        state = torch.from_numpy(state).float().to(device)
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)
        # here make a filter for only possible actions!
        action = dist.sample()

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
    def __init__(self, state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip):
        self.lr = lr
        self.betas = betas
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs

        self.policy = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr, betas=betas,  eps=1e-5) # no eps before!
        self.policy_old = ActorCritic(state_dim, action_dim, n_latent_var).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def update(self, memory):
        # Monte Carlo estimate of state rewards:
        # rewards = []
        # discounted_reward = 0
        # for reward, is_terminal in zip(reversed(memory.rewards), reversed(memory.is_terminals)):
        #     if is_terminal:
        #         discounted_reward = 0
        #     discounted_reward = reward + (self.gamma * discounted_reward)
        #     rewards.insert(0, discounted_reward)

        # Normalizing the rewards:
        rewards = torch.tensor(memory.rewards).to(device)
        rewards = rewards/100

        # convert list to tensor
        old_states = torch.stack(memory.states).to(device).detach()
        old_actions = torch.stack(memory.actions).to(device).detach()
        old_logprobs = torch.stack(memory.logprobs).to(device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        return rewards.mean(), len(rewards)

def exportONNX(model, input_vector, path):
    torch_out = torch.onnx._export(model, input_vector, path+".onnx",  export_params=True)


def main():
    start_time = datetime.datetime.now()
    #stdout.write_file("hallo.txt")
    ############## Hyperparameters ##############
    env_name = "Witches-v0"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n
    render = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval = 50           # print avg reward in the interval
    max_episodes = 500000       # max training episodes
    # TODO DO NOT RESET AFTER FIXED VALUE BUT AT END OF Game
    # THIS DEPENDS IF YOU DO ALLOW TO LEARN THE RULES!
    nu_games        = 5              # max game steps!
    n_latent_var    = 64      # number of variables in hidden layer
    update_timestep = 5      # update policy every n timesteps befor:
    lr              = 0.0003  #in big2game: 0.00025
    gamma           = 0.99
    betas           = (0.9, 0.999)
    K_epochs        = 4                # update policy for K epochs in big2game:nOptEpochs = 5
    eps_clip        = 0.3              # clip parameter for PPO
    random_seed     = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print("Learning Parameters:", lr, nu_games, n_latent_var, update_timestep, K_epochs, eps_clip, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    reward_mean = 0
    wrong_moves = 0
    invalid_moves = 0

    # training loop
    for i_episode in range(1, max_episodes+1):
        for t in range(nu_games):
            timestep += 1
            state = env.reset()
            done  = 0
            while not done:
                # Running policy_old:
                # state has to be right before the AI Plays!
                action = ppo.policy_old.act(state, memory)

                # this should be the reward for the above action
                # this is the new state! when the ai player is again
                state, reward, done, _ = env.step(action)
                if reward==-100:
                    invalid_moves +=1

                # Saving reward and is_terminal:
                memory.rewards.append(reward)
                memory.is_terminals.append(done)

            # update if its time
            if timestep % update_timestep == 0:
                reward_mean, wrong_moves = ppo.update(memory)
                #wrong_moves -= update_timestep*60
                memory.clear_memory()
                timestep = 0

            running_reward += reward


        avg_length += t

        # stop training if avg_reward > solved_reward
        if running_reward > (log_interval*solved_reward):
            print("########## Solved! ##########")
            torch.save(ppo.policy.state_dict(), './PPO_{}.pth'.format(env_name))
            break

        # logging
        if i_episode % log_interval == 0:
            avg_length = int(avg_length/log_interval)
            running_reward = int((running_reward/log_interval))
            #stdout.enable()
            aaa = ('Episode {} \t reward_mean: {:0.5}\t{:0.3} \t invalid_moves: {}\t{}\n'.format(i_episode, reward_mean, (reward_mean*100)-21, invalid_moves, datetime.datetime.now()-start_time))
            print(aaa)
            invalid_moves = 0
            #exportONNX(ppo.policy, torch.rand(180), str(reward_mean))
            with open("hallo.txt", "a") as myfile:
                myfile.write(aaa)
            #stdout.disable()
            running_reward = 0
            avg_length = 0

if __name__ == '__main__':
    main()
