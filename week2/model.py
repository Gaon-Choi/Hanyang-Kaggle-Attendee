import os

import numpy
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

from week2.app import SubmittedApp

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(4, 150)
        self.fc2 = nn.Linear(150, 150)
        self.fc3 = nn.Linear(150, 3)

    def forward(self, x):
        out = torch.nn.functional.relu(self.fc1(x))
        out = self.fc2(out)
        out = self.fc3(out)

        return out

if __name__ == "__main__":
    EPOCH = 10000
    features, labels = load_iris(return_X_y = True)
    x_train, x_test, y_train, y_test = train_test_split(features, labels, random_state = 42)

    # Training
    model = Model()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    epochs = 100

    x_train = Variable(torch.from_numpy(x_train)).float()
    y_train = Variable(torch.from_numpy(y_train)).long()

    for epoch in range(1, EPOCH+1):
        print("Epoch #", epoch)
        y_pred = model(x_train)
        loss = loss_fn(y_pred, y_train)
        print("Loss: ", loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    path = os.path.join('./', "model.pth")
    print("Saving skpt... path: {}".format(path))
    torch.save(model.state_dict(), path)