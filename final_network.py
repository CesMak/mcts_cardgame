# tested with python 3.7.5
from torch.utils.data import Dataset
import ast
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.autograd import Variable

# Links:
# https://www.kaggle.com/danieldagnino/training-a-classifier-with-pytorch
# https://discuss.pytorch.org/t/training-a-card-game-classifier/70625/2

class MyDataSet(Dataset):
    '''A line in the dataset consists of
    [bin_options+bin_on_table+bin_played]+[action_output+[round(bestq, 2)]]
    Input: 180x1
    Output: one-hot encoded 60x1
    The bestq comes from the mcts estimation of how good the played action is
    This values is currently not used.
    '''
    def __init__(self, data_root):
        self.samples = []
        with open(data_root,"r") as f:
            self.samples = [ [ast.literal_eval(ast.literal_eval(elem)[0]), ast.literal_eval(ast.literal_eval(elem)[1])] for elem in f.read().split('\n') if elem]
        print("Number of samples:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Function that returns one input and one output (label)
        # as one dim output use:
        # torch.Tensor(self.samples[idx][0]), torch.Tensor([self.samples[idx][1]].index(1))
        return torch.Tensor(self.samples[idx][0]), torch.Tensor([self.samples[idx][1].index(1)])

class my_model(nn.Module):
    '''
    '''
    def __init__(self, n_in=180, n_middle=120*8, n_out=60):
        super(my_model, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        self.n_middle = n_middle

        self.fc1 = nn.Linear(self.n_in, self.n_middle)
        # TODO
        # insert fully con layer here direclty connect
        # hand with output
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(self.n_middle, self.n_out)
        self.logprob = nn.LogSoftmax(dim=0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.logprob(x)
        return x

def test_trained_model(input_vector, a=None):
    PATH = 'test23.pth'
    input_vector = torch.tensor(input_vector).float()
    print(input_vector)
    net = my_model()
    net.load_state_dict(torch.load(PATH))
    outputs = net(input_vector)
    print("\nNow you should get the class probabilities of these outputs")
    #print(input_vector.shape)
    #print(outputs.shape)
    print(outputs)
    _, predicted = torch.max((outputs), 0)
    print(predicted)
    return predicted
    # predictions = torch.tensor([outputs[i] for i in available_inputs])
    # max_val, idx = torch.max(predictions, 0)
    # print("Possible:", available_inputs)
    # print("Q-Values:", torch.round(predictions * 10**2) / (10**2))
    # print("Best: q :", max_val, available_inputs[idx])
    #return available_inputs[idx]

if __name__ == '__main__':
    from torch.utils.data import DataLoader
    my_data = MyDataSet('actions__.txt')

    my_loader = DataLoader(my_data, batch_size=1, num_workers=0)

    model = my_model()

    criterium = nn.CrossEntropyLoss() #nn.NLLLoss()
    # or use SGD
    optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=1e-4)

    # Taining.
    loss_values = []
    running_loss = 0.0
    epoch        = 0

    for k, (data, target) in enumerate(my_loader):
        # Set gradient to 0.
        optimizer.zero_grad()

        # Get Data: x=data (inputs), y=target (outpus, labels)
        data   = Variable(data,          requires_grad=False) # input
        target = Variable(target.long(), requires_grad=False) # output

        # Outputs: For CrossEntropyLoss you need to have target.shape = [nBatch]
        #recast target one hot to single!
        #dummy, integer_class_batch = target.max(dim = 1)

        # Feed forward.  should be: pred.shape = [nBatch, model.n_out]
        pred = model(data)
        #print("Target shape:", target.shape, "data shape", data.shape)
        #print("Pred shape: ([nBatch, 60])", pred.shape)
        #print("target as int:", integer_class_batch.shape)

        loss = criterium(pred, target.view(-1))

        # Gradient calculation.
        loss.backward()

        # print statistics
        running_loss += loss.item()
        if k % 100 == 99:    # print every 100 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, k + 1, running_loss / 100 ))
            loss_values.append(running_loss / 100 )
            running_loss = 0.0

        # Model weight modification based on the optimizer.
        optimizer.step()
    print(loss_values)

    # Save the results here!
    PATH = 'test23.pth'
    torch.save(model.state_dict(), PATH)
    print("I saved your model params to", PATH)

    # Test for one batch input 180x1
    test_trained_model(data[1])

# So you are training your network to output the probability of of each
# of the 60 moves being the right move. The simplest rule will be to
# play the move that has the largest predicted probability of being right.
