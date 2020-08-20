import os
from week1.app import SubmittedApp
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.linear = nn.Linear(1,1)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred

if __name__ == "__main__":

    # read weight-height.csv
    data = pd.read_csv("weight-height.csv")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'


    # customized model for linear regression
    model = Model()
    model = model.to(device)

    criterion = torch.nn.MSELoss(reduction = "sum")
    optimizer = optim.Adam(model.parameters(), lr=1e-2)


    # checker for the accuracy
    checker = SubmittedApp()

    EPOCH = 100000

    X, y = data['Height'], data['Weight']

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    x_train = torch.from_numpy(data['Height'].values).unsqueeze(dim=1).float()
    y_train = torch.from_numpy(data['Weight'].values).unsqueeze(dim=1).float()

    for epoch in range(EPOCH + 1):
        y_pred = model(x_train)

        loss = criterion(y_pred, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if(epoch % 100 == 0):
            print(f'Epoch: {epoch} | Loss: {loss.item()}')
            print(checker.metric(y_pred, y_train))

    path = os.path.join('./', "model.pth")
    print("Saving skpt... path: {}".format(path))
    torch.save(model.state_dict(), path)