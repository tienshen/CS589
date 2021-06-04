from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms

import numpy as np
import torch.utils.data as utils

# The parts that you should complete are designated as TODO
class NNet(nn.Module):
    def __init__(self):
        super(NNet, self).__init__()
        self.fc1 = nn.Linear(784, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        # Use the rectified-linear activation function over x

        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        return output

def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        print(output.shape)
        print(output.dtype)
        print(target.shape)
        print(target.dtype)
        loss = F.cross_entropy(output, target)
        print(loss)
        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0: #Print loss every 100 batch
            print('Train Epoch: {}\tLoss: {:.6f}'.format(
                epoch, loss.item()))
    accuracy = test(model, device, train_loader)
    return accuracy

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            print(output)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy


def main():
    torch.manual_seed(1)
    np.random.seed(1)
    # Training settings
    use_cuda = True # Switch to False if you only want to use your CPU
    learning_rate = 0.002
    NumEpochs = 15
    batch_size = 32

    device = torch.device("cuda" if use_cuda else "cpu")

    train_X = np.load('../../Data/mnist/X_train.npy')
    train_Y = np.load('../../Data/mnist/y_train.npy')

    test_X = np.load('../../Data/mnist/X_test.npy')
    test_Y = np.load('../../Data/mnist/y_test.npy')

    # train_X = train_X.reshape([-1,1,28,28]) # the data is flatten so we reshape it here to get to the original dimensions of images
    # test_X = test_X.reshape([-1,1,28,28])

    # transform to torch tensors
    tensor_x = torch.tensor(train_X, device=device)
    tensor_y = torch.tensor(train_Y, dtype=torch.long, device=device)

    test_tensor_x = torch.tensor(test_X, device=device)
    test_tensor_y = torch.tensor(test_Y, dtype=torch.long)

    train_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size) # create your dataloader

    test_dataset = utils.TensorDataset(test_tensor_x,test_tensor_y) # create your datset
    test_loader = utils.DataLoader(test_dataset) # create your dataloader if you get a error when loading test data you can set a batch_size here as well like train_dataloader

    model = NNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    t_acc_lst = []
    for epoch in range(NumEpochs):
        train_acc = train(model, device, train_loader, optimizer, epoch)
        print('\nTrain set Accuracy: {:.0f}%\n'.format(train_acc))
        test_acc = test(model, device, test_loader)
        print('\nTest set Accuracy: {:.0f}%\n'.format(test_acc))
        t_acc_lst.append(test_acc)

    torch.save(model.state_dict(), "mnist_nn.pt")


    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.set_title("Accuracy vs Epochs", color='C0')

    ax.plot(t_acc_lst)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    ax.legend()
    plt.show()

if __name__ == '__main__':
    main()
