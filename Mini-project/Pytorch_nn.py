from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import datasets, transforms
import pandas as pd
import numpy as np
import torch.utils.data as utils
from sklearn.model_selection import train_test_split

# The parts that you should complete are designated as TODO
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # self.dropout1 = nn.Dropout2d(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.linear1 = nn.Linear(875, 1200)
        self.linear2 = nn.Linear(1200, 1024)
        self.linear3 = nn.Linear(1024, 768)
        self.linear4 = nn.Linear(768, 576)
        self.linear5 = nn.Linear(576, 206)

    def forward(self, x):
        x = self.linear1(x.float())
        x = F.relu(x)
        x = self.linear2(x.float())
        x = F.relu(x)
        x = self.linear3(x.float())
        x = F.relu(x)
        x = self.linear4(x.float())
        x = F.relu(x)
        x = self.linear5(x.float())
        # Apply softmax to x
        x = F.log_softmax(x, dim=1)
        return x



def train(model, device, train_loader, optimizer, epoch):
    # model = model.float()
    model.train()
    correct = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # target.
        # print(output.shape)
        # print(output.dtype)
        # print(target)
        # print(target.dtype)
        loss = F.binary_cross_entropy_with_logits(output, target)


        loss.backward()
        optimizer.step()
        if batch_idx % 100 == 0: #Print loss every 100 batch
            print('Train Epoch: {}\tLoss: {:.6f}'.format(
                epoch, loss.item()))
    # accuracy = test(model, device, train_loader)
    return loss

def test(model, device, test_loader):
    model.eval()
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            predicted = torch.max(output, 1)
            print(predicted)
            # pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    #         correct += pred.eq(target.view_as(pred)).sum().item()
    #
    # accuracy = 100. * correct / len(test_loader.dataset)

    return 0

def load_files():
    x_train = pd.read_csv("train_features.csv") #, nrows = 1500)
    y_train_scored = pd.read_csv("train_targets_scored.csv")#, nrows  = 1500)
    y_train_nonscored = pd.read_csv("train_targets_nonscored.csv")#, nrows  = 1500)
    x_test = pd.read_csv("test_features.csv")
    return x_train, y_train_scored, y_train_nonscored, x_test

def main():
    torch.manual_seed(1)
    np.random.seed(1)
    # Training settings
    use_cuda = True # Switch to False if you only want to use your CPU
    learning_rate = 0.01
    NumEpochs = 20
    batch_size = 16

    device = torch.device("cuda" if use_cuda else "cpu")

    x_train, y_train_scored, y_train_nonscored, x_test = load_files()
    x_train['cp_type'] = (x_train['cp_type'] == 'trt_cp').values.astype(int)
    x_train['cp_dose'] = (x_train['cp_dose'] == 'D1').values.astype(int)
    x_train = x_train.drop("sig_id", 1)
    y_train_scored = y_train_scored.drop("sig_id", 1)
    y_train_nonscored = y_train_nonscored.drop("sig_id", 1)
    x_train, x_test, y_train, y_test = train_test_split(x_train, y_train_scored, random_state=0, test_size=0.2)
    print(x_train.shape)

    # transform to torch tensors
    tensor_x = torch.tensor(x_train.to_numpy(), device=device)
    tensor_y = torch.tensor(y_train.to_numpy(), dtype=torch.float, device=device)

    test_tensor_x = torch.tensor(x_test.to_numpy(), device=device)
    test_tensor_y = torch.tensor(y_test.to_numpy(), dtype=torch.float)

    train_dataset = utils.TensorDataset(tensor_x,tensor_y) # create your datset
    train_loader = utils.DataLoader(train_dataset, batch_size=batch_size) # create your dataloader

    test_dataset = utils.TensorDataset(test_tensor_x,test_tensor_y) # create your datset
    test_loader = utils.DataLoader(test_dataset) # create your dataloader if you get a error when loading test data you can set a batch_size here as well like train_dataloader

    model = Net().to(device)
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

if __name__ == '__main__':
    main()