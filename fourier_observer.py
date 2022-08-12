"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""
import math
import wandb
wandb.login()
WANDB_API_KEY = '78544c6ed5f52873b1588acd09ead571942d7dfd'

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

from Adam import Adam

torch.manual_seed(0)
np.random.seed(0)

################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class FNO2d(nn.Module):
    def __init__(self, modes1, modes2,  width):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(4, self.width) # input channel is 4: (a(x, y), x1, x2, y)

        self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w0 = nn.Conv2d(self.width, self.width, 1)
        self.w1 = nn.Conv2d(self.width, self.width, 1)
        self.w2 = nn.Conv2d(self.width, self.width, 1)
        self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.fc1 = nn.Linear(self.width, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        x1 = self.conv0(x)
        x2 = self.w0(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv1(x)
        x2 = self.w1(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv2(x)
        x2 = self.w2(x)
        x = x1 + x2
        x = F.gelu(x)

        x1 = self.conv3(x)
        x2 = self.w3(x)
        x = x1 + x2

        x = x[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

################################################################
# configs
################################################################
TRAIN_PATH = '/central/groups/tensorlab/khassibi/fourier_neural_operator/data/planes.mat'
TEST_PATH = '/central/groups/tensorlab/khassibi/fourier_neural_operator/data/planes.mat'
path_name = TRAIN_PATH[64:-4]

batch_size = 20
learning_rate = 0.1

if path_name == 'planes':
    ntrain = 3000 - batch_size + 1
    ntest = 1000 - batch_size + 1
elif path_name == 'planes_channel180_minchan': 
    ntrain = 7500 - batch_size + 1
    ntest = 2500 - batch_size + 1
elif path_name == 'planes_channel180_minchan2': 
    ntrain = 9220 - batch_size + 1
    ntest = 4620 - batch_size + 1

epochs = 100
step_size = 100
gamma = 0.5

modes = 12
width = 32

r = 1
if path_name == 'planes':
    s1 = 768//r
    s2 = 288//r
elif 'planes_channel180_minchan' in path_name: 
    s1 = 32//r
    s2 = 32//r    

wandb.init(
    project="FNO Planes",
    config={
        "model": "FNO2d observer",
        "file name": path_name,
        'python file': 'fourier_observer.py',
        "patches": False,
        "permute": True,
        "TRAIN_PATH": TRAIN_PATH,
        "TEST_PATH": TEST_PATH,
        "ntrain": ntrain,
        "ntest": ntest,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "step_size": step_size,
        "gamma": gamma,
        "modes": modes,
        "width": width,
        "r": r,
        "s1": s1,
        "s2": s2
        })


################################################################
# load data and data normalization
################################################################
idx = torch.randperm(ntrain + ntest)
training_idx = idx[:ntrain]
testing_idx = idx[-ntest:]

reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('P_plane').permute(2,0,1)[training_idx][:,::r,::r][:,:s1,:s2]
x2_train = reader.read_field('V_plane').permute(2,0,1)[training_idx][:,::r,::r][:,:s1,:s2]
y_train = reader.read_field('V_plane').permute(2,0,1)[training_idx][:,::r,::r][:,:s1,:s2]

x_train = x_train[1:]
y_train = y_train[1:]
x2_train = x2_train[:-1]

x_train = torch.stack([x_train, x2_train]).permute(1,2,3,0)

reader.load_file(TEST_PATH)
x_test = reader.read_field('P_plane').permute(2,0,1)[testing_idx][:,::r,::r][:,:s1,:s2]
x2_test = reader.read_field('V_plane').permute(2,0,1)[testing_idx][:,::r,::r][:,:s1,:s2]
y_test = reader.read_field('V_plane').permute(2,0,1)[testing_idx][:,::r,::r][:,:s1,:s2]

x_test = x_test[1:]
y_test = y_test[1:]
x2_test = x2_test[:-1]

x_test = torch.stack([x_test, x2_test]).permute(1,2,3,0)

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain - 1, s1,s2, 2)
x_test = x_test.reshape(ntest - 1, s1,s2, 2)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=batch_size, shuffle=True)
test_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x_test, y_test), batch_size=batch_size, shuffle=False)
n_steps_per_epoch = math.ceil(len(train_loader.dataset) / batch_size)

################################################################
# training and evaluation
################################################################
model = FNO2d(modes, modes, width).cuda()
print(count_params(model))

optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

output_path = '/central/groups/tensorlab/khassibi/fourier_neural_operator/outputs/'
output_path += path_name
output_path += '_observer.mat'

myloss = LpLoss(size_average=False)
y_normalizer.cuda()
for ep in range(epochs):
    model.train()
    t1 = default_timer()
    train_l2 = 0
    for step, (x, y) in enumerate(train_loader):
        x, y = x.cuda(), y.cuda()

        optimizer.zero_grad()
        out = model(x).reshape(batch_size, s1, s2)
        out = y_normalizer.decode(out)
        y = y_normalizer.decode(y)

        loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1))
        loss.backward()

        optimizer.step()
        train_l2 += loss.item()
        metrics = {"train/train_loss": loss.item(), 
                   "train/epoch": (step + 1 + (n_steps_per_epoch * ep)) / n_steps_per_epoch}
        
        if step + 1 < n_steps_per_epoch:
            # 🐝 Log train metrics to wandb 
            wandb.log(metrics)

    scheduler.step()

    model.eval()
    test_l2 = 0.0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.cuda(), y.cuda()

            out = model(x).reshape(batch_size, s1, s2)
            out = y_normalizer.decode(out)

            test_loss = myloss(out.view(batch_size,-1), y.view(batch_size,-1)).item()
            test_l2 += test_loss
            test_metrics = {"test/test_loss": test_loss}
            wandb.log(test_metrics)

    train_l2/= ntrain
    test_l2 /= ntest
    avg_metrics = {"train/avg_train_loss": train_l2,
                   "test/avg_test_loss": test_l2}
    wandb.log(avg_metrics)

    t2 = default_timer()
    print(ep, t2-t1, train_l2, test_l2)

    if ep == epochs - 1:
        # scipy.io.savemat('/central/groups/tensorlab/khassibi/fourier_neural_operator/outputs/planes_preds_wandb.mat', mdict={'x': x.cpu().numpy(), 'pred': out.cpu().numpy(), 'y': y.cpu().numpy(),})
        scipy.io.savemat(output_path, mdict={'x': x.cpu().numpy(), 'pred': out.cpu().numpy(), 'y': y.cpu().numpy(),})

# torch.save(model, "/central/groups/tensorlab/khassibi/fourier_neural_operator/outputs/planes")

################################################################
# making the plots
################################################################
mat = scipy.io.loadmat(output_path)

# Plots
for index in [0, 5, 10, 19]:
    vmin = dat['y'][index, :, :].min()
    vmax = dat['y'][index, :, :].max()
    fig, axes = plt.subplots(nrows=1, ncols=4)
    plt.subplot(1, 3, 1)
    im1 = plt.imshow(dat['x'][index, :, :, 0], cmap='jet', aspect='auto')
    plt.title('Input')
    plt.subplot(1, 3, 2)
    im2 = plt.imshow(dat['y'][index, :, :], cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
    plt.title('True Output')
    plt.subplot(1, 3, 3)
    im3 = plt.imshow(dat['pred'][index, :, :], cmap='jet', aspect='auto', vmin=vmin, vmax=vmax)
    plt.title('Prediction')
    cbar_ax = fig.add_axes([.92, 0.15, 0.04, 0.7])
    fig.colorbar(im3, cax=cbar_ax)

    wandb.log({f"chart_{index}": plt})


wandb.finish()