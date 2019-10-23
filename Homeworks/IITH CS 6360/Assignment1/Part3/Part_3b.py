from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# Training settings
batch_size = 128
test_batch_size = 1000
epochs = 2
lr = 0.08
momentum = 0.2
weight_decay = 0.001

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

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)


def train(epoch):
    model.train()
    total_train_errors = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        
        total_train_errors = total_train_errors + get_total_misclassifications(output, target)
        loss.backward()
        optimizer.step()
        '''
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
        '''
    return total_train_errors / len(train_loader.dataset)


def test():
    total_test_errors = 0
    for batch_idx, (data, target) in enumerate(test_loader):
      output = model(data)
      total_test_errors = total_test_errors + get_total_misclassifications(output, target)
    
    return  total_test_errors / len(test_loader.dataset)


def get_total_misclassifications(softmax_output, target):
  '''
  Return the total classification errors in batch
  '''
  total_correctly_classified = torch.sum(torch.eq(torch.argmax(softmax_output, dim=1), target))
  total_labels = target.shape[0]
  return total_labels - total_correctly_classified.item()
  
train_error = []
test_error = []

for epoch in range(1, epochs + 1):
    train_error.append(train(epoch))
    test_error.append(test())
    print("epoch {}, train error {:.2f}, test error {:.2f}".format(epoch, train_error[-1], test_error[-1]))