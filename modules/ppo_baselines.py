#https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb
#https://stable-baselines.readthedocs.io/en/master/guide/custom_env.html

# Installation:
# pip install stable-baselines[mpi]==2.10.0
# sudo apt install libopenmpi-dev
# pip install mpi4py

import gym
import gym_witches
import os
import matplotlib.pyplot as plt

#1. Check Environment
from stable_baselines.common.env_checker import check_env
from stable_baselines import DQN, PPO2, A2C, ACKTR
from stable_baselines.common.cmd_util import make_vec_env

# for plotting results:
from stable_baselines import results_plotter
from stable_baselines.bench import Monitor
from stable_baselines.results_plotter import load_results

# creating environment
env = gym.make("Witches-v0")

log_dir = "tmp/"
os.makedirs(log_dir, exist_ok=True)

# Create and wrap the environment
env = Monitor(env, log_dir)

# wrap it
env_vec = make_vec_env(lambda: env, n_envs=60)

# If the environment don't follow the interface, an error will be thrown
#check_env(env, warn=True)

model = PPO2('MlpLstmPolicy', env_vec, verbose=1)

time_steps = 1e8
model.learn(int(time_steps))

# export model as onnx:
#1. export params

#2. load params in pytorch:
model.save("tmp/witches")



# LOAD MODEL IN PYTORCH:
# import torch
# import torch.nn as nn
# import torch as th
#
# class PyTorchMlp(nn.Module):
#
#   def __init__(self, n_inputs=4, n_actions=2):
#       nn.Module.__init__(self)
#
#       self.fc1 = nn.Linear(n_inputs, 64)
#       self.fc2 = nn.Linear(64, 64)
#       self.fc3 = nn.Linear(64, n_actions)
#       self.activ_fn = nn.Tanh()
#       self.out_activ = nn.Softmax(dim=0)
#
#   def forward(self, x):
#       x = self.activ_fn(self.fc1(x))
#       x = self.activ_fn(self.fc2(x))
#       x = self.out_activ(self.fc3(x))
#       return x
#
# def copy_mlp_weights(baselines_model):
#   torch_mlp = PyTorchMlp(n_inputs=4, n_actions=2)
#   model_params = baselines_model.get_parameters()
#
#   policy_keys = [key for key in model_params.keys() if "pi" in key]
#   policy_params = [model_params[key] for key in policy_keys]
#
#   for (th_key, pytorch_param), key, policy_param in zip(torch_mlp.named_parameters(), policy_keys, policy_params):
#     param = th.from_numpy(policy_param)
#     #Copies parameters from baselines model to pytorch model
#     print(th_key, key)
#     print(pytorch_param.shape, param.shape, policy_param.shape)
#     pytorch_param.data.copy_(param.data.clone().t())
#
#   return torch_mlp
#
# model_path = "{}/{}.pkl".format('trained_agents/ppo2/', 'CartPole-v1')
#
# baselines_mlp_model = PPO2.load(model_path)
# for key, value in baselines_mlp_model.get_parameters().items():
#   print(key, value.shape)
#
# th_model = copy_mlp_weights(baselines_mlp_model)



# obs = env.reset()
# while True:
#     action, states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     print(rewards, dones)
#     env.render()

print(log_dir)
results_plotter.plot_results([log_dir], time_steps, results_plotter.X_TIMESTEPS, "Witches")
plt.show()



## Output PPO2
## https://stable-baselines.readthedocs.io/en/master/modules/ppo2.html

# -------------------------------------
# | approxkl           | 8.841733e-05 |
# | clipfrac           | 0.0          |
# | ep_len_mean        | 1.4          |   mean episode length
# | ep_reward_mean     | 0.2          |   mean reward per episode
# | explained_variance | -0.0164      |
# | fps                | 2831         |
# | n_updates          | 99           |   number of gradient updates
# | policy_entropy     | 4.0691833    |
# | policy_loss        | -0.001613551 |
# | serial_timesteps   | 12672        |
# | time_elapsed       | 278          |
# | total_timesteps    | 760320       |
# | value_loss         | 0.049821034  |
