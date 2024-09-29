import torch
import numpy as np
from scipy.io import loadmat
from scipy.io import savemat
import matplotlib.pyplot as plt
from UNet_Module import Forward_module

x_observed = loadmat('test_smear.mat')
x_observed = x_observed['I_test_smear'].astype(np.float32)
y_clean = loadmat('test_ideal.mat')
y_clean = y_clean['I_test_ideal'].astype(np.float32)

x_re_conv = loadmat('x_conv.mat')
x_re_conv = x_re_conv['x_conv'].astype(np.float32)
x_re_conv = x_re_conv[0, :]

x_observed_input = torch.tensor(x_observed).unsqueeze(0)
model = Forward_module()

model.load_state_dict(torch.load('best_model_desmear_finetune.pth'))

y_ds = model(x_observed_input)
y_ds = y_ds.detach().numpy()
y_ds = y_ds[0, 0, :]
x_observed = x_observed[0, :]
y_clean = y_clean[0, :]

I_test_desmear = np.expand_dims(y_ds, axis=0)
I_test_desmear = {'I_test_desmear': I_test_desmear}
savemat('I_test_desmear.mat', I_test_desmear)

q = loadmat('q.mat')
q = q['q'].astype(np.float32)
q = q[0, :]

plt.figure()
plt.plot(q, y_clean, label='Ground Truth', linewidth=2, color='red')
plt.plot(q, x_observed, label='Smear', linewidth=1, color='blue',linestyle='-')
plt.plot(q, y_ds, 'o', label='Desmear', markerfacecolor='none', color='black')
plt.legend()
plt.show()

plt.figure()
plt.plot(q, x_re_conv, label='recon', linewidth=2, color='red')
plt.plot(q, x_observed,'o', label='Smear', markerfacecolor='none', color='black')
plt.legend()
plt.show()

