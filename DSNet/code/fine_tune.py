import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from UNet_Module import Combine_Net
import time

num_epochs = 500
learn_rate_clear = 0.00001

x_observed = loadmat('I_test_smear_noise_0025.mat')
x_observed = x_observed['I_test_smear_noise_0025'].astype(np.float32)
x_observed_tensor = torch.Tensor(x_observed).unsqueeze(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = Combine_Net().to(device)
model.clear.load_state_dict(torch.load('best_model_desmear_pretrain.pth'))
criterion = nn.MSELoss()
optimizer = optim.Adam([{'params': model.clear.parameters(), 'lr': learn_rate_clear}])

epoch_list = []
train_loss_list = []
best_loss = 1

start_time = time.time()

for epoch in range(num_epochs):
    epoch_list.append(epoch)

    x_observed_tensor = x_observed_tensor.to(device)

    psf, x_conv, x_clear = model(x_observed_tensor)

    loss = criterion(x_observed_tensor, x_conv)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    train_loss_list.append(loss.item())
    print('epoch:{:>3},  loss:{:>10.6f}'.format(epoch + 1, train_loss_list[-1]))

    if loss.item() < best_loss:
        best_loss = loss.item()
        torch.save(model.clear.state_dict(), "best_model_desmear_finetune_noise_0025.pth")

    if epoch >= 5 and loss.item() <= 1e-5:
        break

end_time = time.time()
fine_tuning_time = end_time - start_time
print(f"Total training time: {fine_tuning_time:.2f} seconds")
plt.plot(epoch_list, train_loss_list, label='train loss')
plt.legend()
plt.show()
