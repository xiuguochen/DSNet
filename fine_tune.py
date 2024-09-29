import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from UNet_Module import Combine_Net
from scipy.io import savemat

num_epochs = 200
learn_rate_clear = 0.00001

x_observed = loadmat('test_smear.mat')
x_observed = x_observed['I_test_smear'].astype(np.float32)
x_observed_tensor = torch.Tensor(x_observed).unsqueeze(0)

# 创建CNN实例、定义损失函数和优化器
model = Combine_Net()
model.clear.load_state_dict(torch.load('best_model_desmear_pretrain.pth'))
criterion = nn.MSELoss()
optimizer = optim.Adam([{'params': model.clear.parameters(), 'lr': learn_rate_clear}])

epoch_list = []
train_loss_list = []
best_loss = 1

for epoch in range(num_epochs):
    epoch_list.append(epoch)
    psf, x_conv, x_clear = model(x_observed_tensor)

    # 计算损失，后向传播
    loss = criterion(x_observed_tensor, x_conv)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss_list.append(loss.item())
    print('epoch:{:>3},  loss:{:>10.6f}'.format(epoch + 1, train_loss_list[-1]))
    # 保存最佳拟合的网络参数
    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.clear.state_dict(), "best_model_desmear_finetune.pth")

    # 判断是否停止
    if epoch >= 5 and loss.item() <= 1e-5:
        break

x_conv = x_conv.detach().numpy()
x_conv = x_conv[0, :, :]
x_conv = {'x_conv': x_conv}
savemat('x_conv.mat', x_conv)

plt.plot(epoch_list, train_loss_list, label='train loss')
plt.legend()
plt.show()
