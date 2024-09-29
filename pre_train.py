import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import loadmat
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split  # 划分训练集和测试集
import matplotlib.pyplot as plt
from UNet_Module import Combine_Net

batch_size = 2
num_epochs = 500
learn_rate_clear = 0.0001

x_observed = loadmat('I_smear.mat')
x_observed = x_observed['I_smear'].astype(np.float32)
y_clean = loadmat('I_ideal.mat')
y_clean = y_clean['I_ideal'].astype(np.float32)
x_observed_train, x_observed_test, y_clean_train, y_clean_test = train_test_split(x_observed, y_clean, test_size=0.2)


# 将训练数据集进行批量处理
# prepare dataset

class DesmearingDataset(Dataset):
    def __init__(self, data, label):
        self.len = data.shape[0]  # shape(多少行，多少列)
        self.x_data = torch.from_numpy(data)
        self.y_data = torch.from_numpy(label)

    def __getitem__(self, index):
        return self.x_data[index].unsqueeze(0), self.y_data[index].unsqueeze(0)

    def __len__(self):
        return self.len


train_dataset = DesmearingDataset(x_observed_train, y_clean_train)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workers 多线程

test_dataset = DesmearingDataset(x_observed_test, y_clean_test)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workers 多线程

# 创建CNN实例、定义损失函数和优化器
model = Combine_Net()
criterion = nn.MSELoss()
optimizer = optim.Adam([{'params': model.clear.parameters(), 'lr': learn_rate_clear}])

# 传入GPU
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# model.to(device)

epoch_list = []
train_loss_list = []
test_loss_list = []


# training cycle forward, backward, update
def train(epoch):
    train_loss = 0.0
    count = 0
    for i, data in enumerate(train_loader, 0):
        inputs, labels = data
        # inputs, labels = inputs.to(device), labels.to(device)
        psf, x_conv, y_pred = model(inputs)

        loss = criterion(y_pred, labels) + criterion(x_conv, inputs)

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
    with torch.no_grad():  # 禁用梯度计算，以减少内存消耗和加快计算速度
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            # inputs, labels = inputs.to(device), labels.to(device)
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
    for epoch in range(num_epochs):
        train(epoch)
        epoch_list.append(epoch)
        test_loss_r = test()
        if test_loss_r < best_loss:
            best_loss = test_loss_r
            torch.save(model.clear.state_dict(), "best_model_desmear_pretrain.pth")
        # if epoch % 20 == 19:
        #     epoch_list.append(epoch)
        #     test()

plt.plot(epoch_list, train_loss_list, label='train loss')
plt.plot(epoch_list, test_loss_list, label='test loss')
plt.legend()
plt.show()
