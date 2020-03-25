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
import os
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

## # TODO:
# Schaue wie big2 hearts aufgebaut ist was sind die hyperparameter was die rewards? (discount factor?)
# Nute dann das hier!

# 1. Dass keine Invalid moves!
# Feed in Inputs of Hand card in output options!
# see: https://discuss.pytorch.org/t/how-to-concatenate-two-layers-using-sefl-add-module/28922/2
# 2. Schau dass je weiter gespielt wird in einer runde desto höher der Reward! (damit bis ans Ende kommt!)
# minimum moves bestes Ergebnis:
# Learning Parameters: 0.00025 5 256 5 4 0.3 (0.9, 0.999)
# Der Reward korrliert nicht mit invalid move?!


# 5, 5, 0.0002 funktioniert ganz gut!
# Bestes ergebnis -0.36 als mean reward (pro Zug)
#Learning Parameters: 0.0003 5 64 5 4 0.3 (0.9, 0.999)
#Episode 50 	 reward_mean: -0.92277	-1.13e+02 	 wrong_moves: 870	0:00:31.805750

# other good stats: with esp=1e-05:
# [1.0001e+04 9.0000e+00 3.5580e+03 3.2070e+03]  (9 games wone in <20k games)
# example of invalid moves: Episode 17050 	 reward_mean: -0.045368	-25.5 	 invalid_moves: 2791	0:36:40.401963
# BEST POLICY SO FAR: +2.5 (in each game) done for 5 games

## TODO:
# Increase the value of entropy coeff to encourage exploration

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

# dummy module to be used in sequential
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
        tmp = ActorMod(state_dim, action_dim, n_latent_var)
        self.action_layer = tmp
         # nn.Sequential(
         #        nn.Linear(state_dim, n_latent_var),
         #        nn.Tanh(),
         #        nn.Linear(n_latent_var, n_latent_var),
         #        nn.Tanh(),
         #        nn.Linear(n_latent_var, action_dim),
         #        # here combine layers!
         #        nn.Softmax(dim=-1)
         #        )

        # critic
        self.value_layer = nn.Sequential(
                nn.Linear(state_dim, n_latent_var),
                nn.PReLU(),#prelu
                nn.Linear(n_latent_var, n_latent_var),
                nn.PReLU(),
                nn.Linear(n_latent_var, 1)
                )

    def forward(self, state_input):
        return torch.tensor(self.act(state_input, None))

    def act(self, state, memory):
        if type(state) is np.ndarray:
            state = torch.from_numpy(state).float().to(device)
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
        rewards = torch.tensor(memory.rewards).to(device)                     # use here memory.rewards
        #rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)  # commented out
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
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.05*dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
        return rewards.mean(), len(rewards)

def exportONNX(model, input_vector, path):
    torch_out = torch.onnx._export(model, input_vector, path+".onnx",  export_params=True)

def getOnnxAction(path, x):
        '''Input:
        x:      240x1 list binary values
        path    *.onnx (with correct model)'''
        ort_session = onnxruntime.InferenceSession(path)
        print(ort_session)
        print(ort_session)
        print(ort_session.get_inputs())
        print(ort_session.get_inputs()[0].name)
        print(np.asarray(x, dtype=np.float32))
        ort_inputs  = {ort_session.get_inputs()[0].name: np.asarray(x, dtype=np.float32)}
        ort_outs    = ort_session.run(None, ort_inputs)
        print(ort_outs)
        return np.asarray(ort_outs)[0]

def testOnnxModel(path):
    env_name = "Witches-v0"
    env = gym.make(env_name)
    for i in range(100):
        done  = 0
        state = env.reset()
        total_games_won = np.zeros(4,)
        while not done:
            print(state)
            action = getOnnxAction(path, state)
            print(action)
            state, reward, done, nu_games_won = env.step(action)
            if reward == -100:
                done = True
        #total_games_won+=nu_games_won
        print(total_games_won)

def test_trained_model(path):
    env_name = "Witches-v0"
    # creating environment
    env = gym.make(env_name)
    ppo_test = PPO(240, 60, 128, 25*1e-7, (0.9, 0.999), 0.99, 5, 0.1)
    memory = Memory()
    ppo_test.policy_old.load_state_dict(torch.load(path))
    total_games_won = np.zeros(4,)
    total_nu_of_wrong_moves = 0
    max_games               = 1000
    #torch.onnx.export(ppo_test.policy, torch.rand(240), "onnx_model_name.onnx")

    # Plays 100 games (one game is finished after 70 Points)
    # If an invalid move is played, the ai has to choose again! ??? What is exactly done in this case?
    # Good Stats: [163. 448. 193. 196.]
    while np.sum(total_games_won)<max_games:
        done  = False
        state = env.reset()
        while not done:
            action = ppo_test.policy_old.act(state, memory)
            state, reward, done, nu_games_won= env.step(action)
            if reward==-100:
                total_nu_of_wrong_moves+=1
        total_games_won +=nu_games_won
    print(total_games_won, "invalid_moves:", total_nu_of_wrong_moves)

def main():
    start_time = datetime.datetime.now()
    #stdout.write_file("hallo.txt")
    ############## Hyperparameters ##############
    env_name = "Witches-v0"
    # creating environment
    env = gym.make(env_name)
    state_dim = env.observation_space.n
    action_dim = env.action_space.n

    log_path      = "logging.txt"
    try:
        os.remove(os.getcwd()+"/"+log_path)
    except:
        print("No Logging to be removed!")
    render        = False
    solved_reward = 230         # stop training if avg_reward > solved_reward
    log_interval  = 50           # print avg reward in the interval
    max_episodes  = 500000       # max training episodes
    # TODO DO NOT RESET AFTER FIXED VALUE BUT AT END OF Game
    # THIS DEPENDS IF YOU DO ALLOW TO LEARN THE RULES!
    nu_games        = 5             # max game steps!
    n_latent_var    = 64            # number of variables in hidden layer
    update_timestep = 5             # update policy every n timesteps befor:
    lr              = 25*1e-7       # in big2game: 25*1e-5
    gamma           = 0.99
    betas           = (0.9, 0.999)
    K_epochs        = 5               # update policy for K epochs in big2game:nOptEpochs = 5  typical 3 - 10 is the number of passes through the experience buffer during gradient descent.
    eps_clip        = 0.1             # clip parameter for PPO Setting this value small will result in more stable updates, but will also slow the training process.
    random_seed     = None
    #############################################

    if random_seed:
        torch.manual_seed(random_seed)
        env.seed(random_seed)

    memory = Memory()
    ppo = PPO(state_dim, action_dim, n_latent_var, lr, betas, gamma, K_epochs, eps_clip)
    print("Learning Parameters:", lr, nu_games, n_latent_var, "Update_timestep", update_timestep, K_epochs, eps_clip, betas)

    # logging variables
    running_reward = 0
    avg_length = 0
    timestep = 0
    reward_mean = 0
    wrong_moves = 0
    invalid_moves = 0
    total_number_of_games_played = 0
    total_rewards  = 0
    total_games_won = np.zeros(4,)

    # training loop
    for i_episode in range(1, max_episodes+1):
        timestep += 1
        state = env.reset()
        done  = 0
        while not done:
            # Running policy_old:
            # state has to be right before the AI Plays!
            action = ppo.policy_old.act(state, memory)

            # this should be the reward for the above action
            # this is the new state! when the ai player is again
            state, reward, done, nu_games_won = env.step(action)
            if reward==-100:
                invalid_moves +=1

            # if done and invalid_moves==0:
            #     print("Export ONNX:::")
            #     exportONNX(ppo.policy, torch.rand(240), str(reward_mean))

            total_rewards += reward
            memory.rewards.append(reward)
            memory.is_terminals.append(done)

        total_number_of_games_played+=1
        total_games_won +=nu_games_won

        # update if its time
        if timestep % update_timestep == 0:
            reward_mean, wrong_moves = ppo.update(memory)
            memory.clear_memory()
            timestep = 0

        # logging
        if i_episode % log_interval == 0:
            total_reward_per_game_positive = total_rewards/log_interval
            per_game_reward = total_reward_per_game_positive-15*21
            games_won = str(np.array2string(total_games_won))
            # total_rewards per game should be maximized!!!!
            aaa = ('Game ,{:07d}, reward ,{:0.5}, invalid_moves ,{:4.4}, games_won ,{},  Time ,{},\n'.format(total_number_of_games_played, per_game_reward, invalid_moves/log_interval, games_won, datetime.datetime.now()-start_time))
            print(aaa)
            if total_games_won[1]>40:
                 print("Export ONNX:::")
                 torch.save(ppo.policy.state_dict(), './PPO_{}_{}.pth'.format(env_name, total_games_won[1]))
                 #exportONNX(ppo.policy, torch.rand(240), str(per_game_reward))
            invalid_moves = 0
            total_rewards = 0
            total_games_won = np.zeros(4,)
            with open(log_path, "a") as myfile:
                myfile.write(aaa)

if __name__ == '__main__':
    #main()
    test_trained_model("PPO_Witches-v0_41.0222.pth")# PPO_Witches-v0.
    #testOnnxModel("onnx_model_name.onnx") #-4.5.onnx onnx_model_name.onnx
