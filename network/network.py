# tested with python 3.7.5
import torch
print(torch.__version__)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision # dataset
import torchvision.transforms as transforms


class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(3, 6, 5)
		self.pool = nn.MaxPool2d(2, 2)
		self.conv2 = nn.Conv2d(6, 16, 5)
		self.fc1 = nn.Linear(16 * 5 * 5, 120)
		self.fc2 = nn.Linear(120, 84)
		self.fc3 = nn.Linear(84, 10)

	def forward(self, x):
		x = self.pool(F.relu(self.conv1(x)))
		x = self.pool(F.relu(self.conv2(x)))
		x = x.view(-1, 16 * 5 * 5)
		x = F.relu(self.fc1(x))
		x = F.relu(self.fc2(x))
		x = self.fc3(x)
		return x

net = Net()

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)



transform = transforms.Compose(
	[transforms.ToTensor(),
	 transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
										download=True, transform=transform)
print(trainset)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
										  shuffle=True, num_workers=2)

#Training:
for epoch in range(2):  # loop over the dataset multiple times

	running_loss = 0.0
	for i, data in enumerate(trainloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = net(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		running_loss += loss.item()
		if i % 2000 == 1999:    # print every 2000 mini-batches
			print('[%d, %5d] loss: %.3f' %
				  (epoch + 1, i + 1, running_loss / 2000))
			running_loss = 0.0

print('Finished Training')


# Save the result:
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)




# Trainloader:
# torch.utils.data.DataLoader is an iterator which provides all these features. Parameters used below should be clear. One parameter of interest is collate_fn. You can specify how exactly the samples need to be batched using collate_fn. However, default collate should work fine for most use cases.
#
# dataloader = DataLoader(transformed_dataset, batch_size=4,
#                         shuffle=True, num_workers=4)


# class Net(nn.Module):
# 	def __init__(self, input_size, output_size):
# 		super(Net, self).__init__()
# 		self.linear1 = nn.Linear(input_size, 120)
# 		self.linear2 = nn.Linear(120, 120)
# 		self.linear3 = nn.Linear(120, 120)
# 		self.linear4 = nn.Linear(120, output_size)
# 		self.relu    = nn.ReLU(inplace=True)
# 		self.dropout = nn.Dropout(0.15)
# 		self.softmax = nn.Softmax()
# 	def forward(self, x):
# 		x = self.linear1(x)
# 		x = self.relu(x)
# 		x = self.dropout(x)
# 		# F.relu()
# 		x = self.linear4(x)
# 		x = self.softmax(x)
# 		return x
