import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from UNet_Module import Combine_Net
import time

batch_size = 90
num_epochs = 500
learn_rate_clear = 0.0001

x_observed = loadmat('I_smear_100_sigma2.mat')
x_observed = x_observed['I_smear'].astype(np.float32)
y_clean = loadmat('I_ideal_100_sigma2.mat')
y_clean = y_clean['I_ideal'].astype(np.float32)
x_observed_train, x_observed_test, y_clean_train, y_clean_test = train_test_split(x_observed, y_clean, test_size=0.1)


class DesmearingDataset(Dataset):
    def __init__(self, data, label):
        self.len = data.shape[0]
        self.x_data = torch.from_numpy(data)
        self.y_data = torch.from_numpy(label)

    def __getitem__(self, index):
        return self.x_data[index].unsqueeze(0), self.y_data[index].unsqueeze(0)

    def __len__(self):
        return self.len


train_dataset = DesmearingDataset(x_observed_train, y_clean_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

test_dataset = DesmearingDataset(x_observed_test, y_clean_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Combine_Net().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam([{'params': model.clear.parameters(), 'lr': learn_rate_clear}])

epoch_list = []
train_loss_list = []
test_loss_list = []


def train(epoch):
    train_loss = 0.0
    count = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        psf, x_conv, y_pred = model(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        count += 1

    train_loss_list.append(train_loss / count)
    print("epoch:", epoch, "train loss:", train_loss / count, end=',')


def test():
    global batch_size
    test_loss = 0.0
    count = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            psf, x_conv, y_pred = model(inputs)
            loss = criterion(y_pred, labels)
            loss = loss.item()
            test_loss += loss
            count += 1
        test_loss_list.append(test_loss / count)
        print("test loss:", test_loss / count)
        return test_loss / count


if __name__ == '__main__':
    best_loss = 0.5
    start_time = time.time()
    for epoch in range(num_epochs):
        train(epoch)
        epoch_list.append(epoch)
        test_loss_r = test()
        if test_loss_r < best_loss:
            best_loss = test_loss_r
            torch.save(model.clear.state_dict(), "best_model_desmear_pretrain.pth")
    end_time = time.time()
    training_time = end_time - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    plt.plot(epoch_list, train_loss_list, label='train loss')
    plt.plot(epoch_list, test_loss_list, label='test loss')
    plt.legend()
    plt.show()
