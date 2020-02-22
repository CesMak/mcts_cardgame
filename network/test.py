# tested with python 3.7.5
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.utils.data as data

# Links:
# https://www.kaggle.com/danieldagnino/training-a-classifier-with-pytorch

# First we create the point that we are going to use for the classifier.
# We create n_points points for four classes of points center at [0,0],
# [0,2], [2,0] and [2,2] with a deviation from the center that follows a
# Gaussian distribution with a standar deviation of sigma.

n_points = 20000
points = np.zeros((n_points,2))   # x, y
target = np.zeros((n_points,1))   # label
sigma = 0.5
for k in range(n_points):
    # Random selection of one class with 25% of probability per class.
    random = np.random.rand()
    if random<0.25:
        center = np.array([0,0])
        target[k,0] = 0   # This points are labeled 0.
    elif random<0.5:
        center = np.array([2,2])
        target[k,0] = 1   # This points are labeled 1.
    elif random<0.75:
        center = np.array([2,0])
        target[k,0] = 2   # This points are labeled 2.
    else:
        center = np.array([0,2])
        target[k,0] = 3   # This points are labeled 3.
    gaussian01_2d = np.random.randn(1,2)
    points[k,:] = center + sigma*gaussian01_2d

# Now, we write all the points in a file.
points_and_labels = np.concatenate((points,target),axis=1)   # 1st, 2nd, 3nd column --> x,y, label
pd.DataFrame(points_and_labels).to_csv('clas.csv',index=False)


# Here, we start properly the classifier.

# We read the dataset and create an iterable.
class my_points(data.Dataset):
    def __init__(self, filename):
        pd_data = pd.read_csv(filename).values   # Read data file.
        self.data = pd_data[:,0:2]   # 1st and 2nd columns --> x,y
        self.target = pd_data[:,2:]  # 3nd column --> label
        self.n_samples = self.data.shape[0]

    def __len__(self):   # Length of the dataset.
        return self.n_samples

    def __getitem__(self, index):   # Function that returns one point and one label.
        return torch.Tensor(self.data[index]), torch.Tensor(self.target[index])



# We create the dataloader.
my_data = my_points('clas.csv')
batch_size = 1
my_loader = data.DataLoader(my_data, batch_size=batch_size, num_workers=0)

# We build a simple model with the inputs and one output layer.
class my_model(nn.Module):
    def __init__(self,n_in=2, n_hidden=10, n_out=4):
        super(my_model,self).__init__()
        self.n_in  = n_in
        self.n_out = n_out

        self.linearlinear = nn.Sequential(
            nn.Linear(self.n_in,self.n_out,bias=True),   # Hidden layer.
            )
        self.logprob = nn.LogSoftmax(dim=1)                 # -Log(Softmax probability).

    def forward(self,x):
        x = self.linearlinear(x)
        print("In forward:", x.shape)
        x = self.logprob(x)
        print("In forward:", x.shape)
        return x
# Now, we create the mode, the loss function or criterium and the optimizer
# that we are going to use to minimize the loss.

# Model.
model = my_model()

# Negative log likelihood loss.
criterium = nn.NLLLoss()

# Adam optimizer with learning rate 0.1 and L2 regularization with weight 1e-4.
optimizer = torch.optim.Adam(model.parameters(),lr=0.1,weight_decay=1e-4)

# Taining.
for k, (data, target) in enumerate(my_loader):
    # Definition of inputs as variables for the net.
    # requires_grad is set False because we do not need to compute the
    # derivative of the inputs.
    print("\nEpoch:", k)
    print(data)
    print(target)
    data   = Variable(data,requires_grad=False)
    target = Variable(target.long(),requires_grad=False)

    # Set gradient to 0.
    optimizer.zero_grad()
    # Feed forward.
    pred = model(data)

    print("Pred.shape:", pred.shape) ### torch.Size([5, 4])  4 ausägnge
    print("target.shape:", target.shape) ### torch.Size([5, 1])
    print(target.view(-1).shape)

    # Loss calculation.
    loss = criterium(pred, target.view(-1))
    # Gradient calculation.
    loss.backward()

    # Print loss every 10 iterations.
    if k%10==0:
        print('Loss {:.4f} at iter {:d}'.format(loss.item(),k))

    # Model weight modification based on the optimizer.
    optimizer.step()

print("Predictions")
print(pred.exp())
pred = pred.exp().detach()     # exp of the log prob = probability.
print(pred)
a, index = torch.max(pred,1)   # index of the class with maximum probability.
print(a, index)
