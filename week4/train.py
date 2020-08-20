import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from week4.model import ImageRNN

if __name__ == "__main__":
    # Importing the dataset
    BATCH_SIZE = 64

    # list all transformations
    transform = transforms.Compose([transforms.ToTensor()])

    # download and load training dataset
    trainset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)

    # download and load testing dataset
    testset = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

    # parameters
    N_STEPS = 28
    N_INPUTS = 28
    N_NEURONS = 150
    N_OUTPUTS = 10
    N_EPHOCS = 10

    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model instance
    model = ImageRNN(BATCH_SIZE, N_STEPS, N_INPUTS, N_NEURONS, N_OUTPUTS)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    def get_accuracy(logit, target, batch_size):
        ''' Obtain accuracy for training round '''
        corrects = (torch.max(logit, 1)[1].view(target.size()).data == target.data).sum()
        accuracy = 100.0 * corrects / batch_size
        return accuracy.item()


    for epoch in range(N_EPHOCS):  # loop over the dataset multiple times
        train_running_loss = 0.0
        train_acc = 0.0
        model.train()

        # TRAINING ROUND
        for i, data in enumerate(trainloader):
            # zero the parameter gradients
            optimizer.zero_grad()

            # reset hidden states
            model.hidden = model.init_hidden()

            # get the inputs
            inputs, labels = data
            inputs = inputs.view(-1, 28, 28)

            # forward + backward + optimize
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_running_loss += loss.detach().item()
            train_acc += get_accuracy(outputs, labels, BATCH_SIZE)

        model.eval()
        print('Epoch:  %d | Loss: %.4f | Train Accuracy: %.2f'
              % (epoch, train_running_loss / i, train_acc / i))

    # Calculate test accuracy
    test_acc = 0.0
    for i, data in enumerate(testloader, 0):
        inputs, labels = data
        inputs = inputs.view(-1, 28, 28)

        outputs = model(inputs)

        test_acc += get_accuracy(outputs, labels, BATCH_SIZE)

    print('Test Accuracy: %.2f' % (test_acc / i))

    path = os.path.join('./', "model.pth")
    print("Saving skpt... path: {}".format(path))
    torch.save(model.state_dict(), path)