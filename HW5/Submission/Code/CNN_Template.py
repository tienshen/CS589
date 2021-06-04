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
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(9216,128)
        self.linear2 = nn.Linear(128,10)

    def forward(self, x):
        # print(x.shape)
        x = self.conv1(x)
        # print(x.shape)
        # Use the rectified-linear activation function over x
        x = F.relu(x)
        x = self.conv2(x)
        # print(x.shape)
        x = F.relu(x)
        # Run max pooling over x
        x = F.max_pool2d(x, 2)
        # print(x.shape)
        # Pass data through dropout1
        x = self.dropout1(x)
        # Flatten x with start_dim=1
        x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.linear1(x)
        # print(x.shape)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.linear2(x)
        # print(x.shape)
        # Apply softmax to x
        # x = F.log_softmax(x, dim=1)
        return x



def train(model, device, train_loader, optimizer, epoch):
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.cross_entropy(output, target)
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
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    accuracy = 100. * correct / len(test_loader.dataset)

    return accuracy


def main():
    torch.manual_seed(1)
    np.random.seed(1)
    # Training settings
    use_cuda = True # Switch to False if you only want to use your CPU
    learning_rate = 0.01
    NumEpochs = 10
    batch_size = 32

    device = torch.device("cuda" if use_cuda else "cpu")

    train_X = np.load('../../Data/X_train.npy')
    train_Y = np.load('../../Data/y_train.npy')

    test_X = np.load('../../Data/X_test.npy')
    test_Y = np.load('../../Data/y_test.npy')

    train_X = train_X.reshape([-1,1,28,28]) # the data is flatten so we reshape it here to get to the original dimensions of images
    test_X = test_X.reshape([-1,1,28,28])

    # transform to torch tensors
    tensor_x = torch.tensor(train_X, device=device)
    tensor_y = torch.tensor(train_Y, dtype=torch.long, device=device)

    test_tensor_x = torch.tensor(test_X, device=device)
    test_tensor_y = torch.tensor(test_Y, dtype=torch.long)

    train_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size) # create your dataloader

    test_dataset = utils.TensorDataset(test_tensor_x,test_tensor_y) # create your datset
    test_loader = utils.DataLoader(test_dataset) # create your dataloader if you get a error when loading test data you can set a batch_size here as well like train_dataloader

    model = ConvNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    train_acc_lst = []
    test_acc_lst = []
    for epoch in range(NumEpochs):
        train_acc = train(model, device, train_loader, optimizer, epoch)
        train_acc_lst.append(train_acc)
        print('\nTrain set Accuracy: {:.1f}%\n'.format(train_acc))
        test_acc = test(model, device, test_loader)
        test_acc_lst.append(test_acc)
        print('\nTest set Accuracy: {:.1f}%\n'.format(test_acc))
        for param in model.parameters():
            print(param.shape)

    print(test_acc_lst)
    torch.save(model.state_dict(), "mnist_cnn.pt")


    fig, ax = plt.subplots()  # Create a figure containing a single axes.
    ax.set_title("Accuracy vs Epochs", color='C0')

    ax.plot(test_acc_lst)
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    ax.legend()
    plt.show()


if __name__ == '__main__':
    main()
