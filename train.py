# tested with python 3.7.5
import pickle
import ast
import datetime
import bz2 # pip install compress-pickle

# torch
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.autograd import Variable

import matplotlib.pyplot as plt

# For exporting the model:
import torch.onnx
import onnx
import onnxruntime

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
    def __init__(self, data_root, use_pickle=0, pickle_name="data/actions__.pkl"):
        self.samples = []
        self.pickle_name = pickle_name
        if use_pickle:
            infile = bz2.open(self.pickle_name ,'rb')
            self.samples = pickle.load(infile, encoding='bytes')
        else:
            with open(data_root, "r") as f:
                self.samples = [ [ast.literal_eval(ast.literal_eval(elem)[0]), ast.literal_eval(ast.literal_eval(elem)[1])[0:60].index(1)] for elem in f.read().split('\n') if elem]
        print("Read in samples:\t"+str(len(self.samples)))
        print("One sample:")
        print(self.samples[0])

    def __convert2pickle__(self):
        # Saves a lot of storage:
        sfile = bz2.BZ2File(self.pickle_name, 'w')
        pickle.dump(self.samples, sfile, protocol=2)
        sfile.close()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Function that returns one input and one output (label)
        return torch.Tensor(self.samples[idx][0]), torch.Tensor([self.samples[idx][1]])

class my_model(nn.Module):
    def __init__(self,n_in=180, n_hidden=60, n_out=60):
        super(my_model,self).__init__()
        self.n_in  = n_in
        self.n_out = n_out
        # Hidden layer.
        self.linearlinear = nn.Sequential(nn.Linear(self.n_in,  self.n_out, bias=True),)
        self.logprob = nn.LogSoftmax(dim=0)                 # -Log(Softmax probability).

    def forward(self,x):
        x = self.linearlinear(x)
        x = self.logprob(x)
        return x

class my_model_2(nn.Module):
    '''
    '''
    def __init__(self, n_in=180, n_hidden=1, n_out=60):
        super(my_model, self).__init__()
        self.n_in  = n_in
        self.n_out = n_out

        self.fc1 = nn.Linear(self.n_in, 120)
        # TODO
        # insert fully con layer here direclty connect
        # hand with output
        self.relu1 = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(120, self.n_out)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        return x

#### General Functions
def conver2pickle(pkl_name):
    from torch.utils.data import DataLoader
    my_data = MyDataSet('data/actions__.txt', use_pickle=1, pickle_name=pkl_name)
    my_data.__convert2pickle__()

def load_data(pkl_name, batch_size_):
    from torch.utils.data import DataLoader
    my_data = MyDataSet('data/actions__.txt', use_pickle=1, pickle_name=pkl_name)
    return DataLoader(my_data, batch_size=batch_size_, num_workers=0)

def to_numpy(tensor):
    return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

def test_onnx(path, x):
    # onnx_model = onnx.load(path)
    # onnx.checker.check_model(onnx_model)
    # print("loaded onxx model", onnx_model)
    print("I will now check your onnx model using onnx")

    ort_session = onnxruntime.InferenceSession(path)
    print(ort_session)
    print(ort_session.get_inputs())
    print(ort_session.get_inputs()[0].name)

    # compute ONNX Runtime output prediction
    print("I will now test your model!")
    print("Inputs:", x)
    print("Input type:", type(x))
    y = to_numpy(x)
    print(type(y))
    print(y)
    ort_inputs = {ort_session.get_inputs()[0].name: y}
    ort_outs = ort_session.run(None, ort_inputs)
    print(ort_outs)

def save_model(model_to_save, path, start_time, input_vector=None):
    print("Finished Training in:\t"+str(datetime.datetime.now()-start_time))
    torch.save(model_to_save.state_dict(), path)
    print("I saved your model to:\t"+path)

    # Export the model
    # see: https://pytorch.org/tutorials/advanced/super_resolution_with_caffe2.html
    # and: https://pytorch.org/tutorials/advanced/super_resolution_with_onnxruntime.html
    print("I now save your onnx model with parameters!")
    torch_out = torch.onnx._export(model_to_save, input_vector, path+".onnx",  export_params=True)

def test_trained_model(input_vector, path):
    # input vector: as list 180x1 0...1...0...1
    input_vector = torch.tensor(input_vector[0]).float()
    #print("Shape of Vector:\t"+str(input_vector.shape))

    net = my_model()
    net.load_state_dict(torch.load(path))
    outputs = net(input_vector)
    #print("Shape of Outpus:\t"+str(outputs.shape))
    print("Outputs: using pytorch:")
    print(outputs)
    a, predicted = torch.max(outputs/100, 0)
    print("predicted:", predicted)
    return predicted


if __name__ == '__main__':
    my_loader = load_data("data/actions__.pkl", batch_size_=10)
    path      = "data/model.pth"

    model = my_model()
    criterium = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)
    #optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Training
    start_time   = datetime.datetime.now()
    running_loss = 0.0
    epoch        = 0
    for k, (data, target) in enumerate(my_loader):

        data   = Variable(data,          requires_grad=False) # input
        target = Variable(target.long(), requires_grad=False) # output

        # Set gradient to 0.
        optimizer.zero_grad()

        # Feed forward.
        pred = model(data)

        # Note for CrossEntropyLoss and NLLLoss target must be a label=one number not an array etc.
        loss = criterium(pred, target.view(-1))

        # Gradient calculation.
        loss.backward()

        # print statistics
        running_loss += loss.item()
        if k % 100 == 99:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' % (epoch + 1, k + 1, running_loss ))
            running_loss = 0
        # Model weight modification based on the optimizer.
        optimizer.step()

    print(data[0])
    save_model(model, path, start_time, input_vector=data[0])
    test_onnx(path+".onnx", data[0])
    test_trained_model([data[0]], path)
