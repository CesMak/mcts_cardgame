# tested with python 3.7.5
from torch.utils.data import Dataset
import ast
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable
import tkinter

# Links:
# https://www.kaggle.com/danieldagnino/training-a-classifier-with-pytorch

class my_model(nn.Module):
    def __init__(self,n_in=180, n_hidden=60, n_out=60):
        super(my_model,self).__init__()
        self.n_in  = n_in
        self.n_out = n_out

        self.linearlinear = nn.Sequential(
            nn.Linear(self.n_in,  self.n_out, bias=True),   # Hidden layer.
            )
        self.logprob = nn.LogSoftmax(dim=0)                 # -Log(Softmax probability).

    def forward(self,x):
        x = self.linearlinear(x)
        x = self.logprob(x)
        return x

def test_trained_model(input_vector, e=None):
    #print(input_vector)
    PATH = 'network/abc.pth'
    #torch.save(model.state_dict(), PATH)

    # testing:
    input_vector = torch.tensor(input_vector).float()
    #print(input_vector)
    #print(input_vector.shape)
    net = my_model()
    net.load_state_dict(torch.load(PATH))
    outputs = net(input_vector)
    #print(outputs.shape) # 60x1 mit welcher
    #print("Mit welcher warscheinlichkeit soll welcher index gespielt werden!")
    #print((outputs)/100)
    _, predicted = torch.max((outputs)/100, 0)
    return predicted

class TESNamesDataset(Dataset):
    def __init__(self, data_root):
        self.samples = []
        with open(data_root,"r") as f:
            self.samples = [ [ast.literal_eval(ast.literal_eval(elem)[0]), ast.literal_eval(ast.literal_eval(elem)[1])] for elem in f.read().split('\n') if elem]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Function that returns one input and one output (label)
        return torch.Tensor(self.samples[idx][0]), torch.Tensor([self.samples[idx][1].index(1)])



if __name__ == '__main__':
    from torch.utils.data import DataLoader
    my_data = TESNamesDataset('actions__.txt')
    my_loader = DataLoader(my_data, batch_size=1, num_workers=0)

    model = my_model()
    print("Learnable params:\n,", list(model.parameters()))
    criterium = nn.NLLLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)

    # Taining.
    loss_values = []
    running_loss = 0.0
    epoch        = 0
    for k, (data, target) in enumerate(my_loader):
        #print(data.view(-1).shape) # 1x60
        data   = Variable(data,          requires_grad=False) # input
        target = Variable(target.long(), requires_grad=False) # output

        #squeeze the target here?!
        # s your target has a channel dimension of 1, which is not needed using nn.CrossEntropyLoss or nn.NLLLoss.
        #target = target.squeeze(1)

        # Set gradient to 0.
        optimizer.zero_grad()
        # Feed forward.
        pred = model(data)

        loss = criterium(pred, target.view(-1))

        # Gradient calculation.
        loss.backward()

        # print statistics
        running_loss += loss.item()
        print(loss.item())
        if k % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, k + 1, running_loss / 100 ))
            loss_values.append(running_loss / 100 )
            running_loss = 0.0


        # Model weight modification based on the optimizer.
        optimizer.step()

    print(loss_values)
    # Save the results here!
    PATH = 'abc.pth'
    torch.save(model.state_dict(), PATH)
    print("I saved your model params to", PATH)

    plt.plot(loss_values)
    plt.show()
