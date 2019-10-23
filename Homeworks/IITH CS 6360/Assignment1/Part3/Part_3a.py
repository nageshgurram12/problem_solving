from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Training settings
batch_size = 64
test_batch_size = 1000
epochs = 10
lr = 0.1

seed = 1
log_interval = 10

torch.manual_seed(seed)

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=test_batch_size, shuffle=True)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x)

model = Net()

optimizer = optim.SGD(model.parameters(), lr=lr)


def train(epoch):
    model.train()
    total_train_loss = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        
        total_train_loss = total_train_loss + loss
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            
    return total_train_loss/batch_idx



def test():
    total_test_loss = 0
    for batch_idx, (data, target) in enumerate(test_loader):
      output = model(data)
      loss = F.nll_loss(output, target)
      total_test_loss = total_test_loss + loss
    
    return total_test_loss / batch_idx

train_loss = []
test_loss = []

for epoch in range(1, epochs + 1):
    train_loss.append(train(epoch))
    test_loss.append(test())
    #print("epoch {}, train loss {}, test loss {}".format(epoch, train_loss[-1], test_loss[-1]))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(range(1, epochs+1), train_loss, color='blue', label="Train loss")
ax.plot(range(1, epochs+1), test_loss, color='red', label="Test loss")
ax.set(xlabel="Epochs", ylabel="Loss")
plt.legend()
plt.show()