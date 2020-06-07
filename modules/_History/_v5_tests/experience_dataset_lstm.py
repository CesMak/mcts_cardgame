import torch
from torch.utils import data

class ExperienceDatasetLSTM(data.Dataset):

  'Characterizes a dataset for PyTorch'
  def __init__(self, states, actions, logprobs, rewards):
        'Initialization'
        self.states = states
        self.actions = actions
        self.logprobs = logprobs
        self.rewards = rewards

  def __len__(self):
        'Denotes the total number of samples'
        return len(self.actions)

  def __getitem__(self, index):
        'Generates one sample of data'
        return [self.states[index], self.actions[index], self.logprobs[index], self.rewards[index]]

def custom_collate(batch):
    # See also orginal - was significantly changed....
    #memory.states, memory.actions, memory.logprobs, rewards

    states_batch, actions_batch, logprobs_batch, rewards_batch = zip(*batch)
    #types are all tuple e.g. states_batch -> convert to tensor!

    states   = torch.tensor(states_batch).detach()
    actions  = torch.tensor(actions_batch).detach()
    logprobs = torch.tensor(logprobs_batch).detach()
    rewards  = torch.tensor(rewards_batch).detach()

    return [states, actions, logprobs, rewards]
