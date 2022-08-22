import wandb
wandb.login()
WANDB_API_KEY = '78544c6ed5f52873b1588acd09ead571942d7dfd'

import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

import seaborn as sns
import pandas as pd

import math

import operator
from functools import reduce
from functools import partial

from timeit import default_timer
from utilities3 import *

from Adam import Adam

TRAIN_PATH = '/central/groups/tensorlab/khassibi/fourier_neural_operator/data/planes.mat'
TEST_PATH = '/central/groups/tensorlab/khassibi/fourier_neural_operator/data/planes.mat'

ntrain = 3000
ntest = 1000

r = 8
s1 = 768//r
s2 = 288//r

idx = torch.arange(ntrain + ntest)
training_idx = idx[:ntrain]
testing_idx = idx[-ntest:]

reader = MatReader(TRAIN_PATH)
y_train = reader.read_field('V_plane')
y_train = y_train.permute(2,0,1)[training_idx][:,::r,::r][:,:s1,:s2]

reader.load_file(TEST_PATH)
y_test = reader.read_field('V_plane').permute(2,0,1)[testing_idx][:,::r,::r][:,:s1,:s2]

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

mean_train_velocity = []
for i in range(len(y_train)):
    sum = 0
    for j in range(len(y_train[i])):
        for k in range(len(y_train[i][j])):
            sum += x_train[i][j][k]
    sum /= len(y_train[i]) * len(y_train[i][j])
    mean_train_velocity.append(sum)

mean_test_velocity = []
for i in range(len(y_test)):
    sum = 0
    for j in range(len(y_test[i])):
        for k in range(len(y_test[i][j])):
            sum += x_train[i][j][k]
    sum /= len(y_test[i]) * len(y_test[i][j])
    mean_test_velocity.append(sum)

# make dataset
d = {'train': mean_train_velocity, 'test': mean_test_velocity}
df = pd.DataFrame(data=d)

wandb.init(
    project="FNO Planes",
    config={
        "model": "joint probability"
    }
)

  
# draw jointplot with
# kde kind
g = sns.jointplot(x = "train", y = "test", kind = "kde", data = df)
wandb.log({f"chart_{index}": plt})
# Show the plot
plt.show()
  
# FileData = load('/central/groups/tensorlab/khassibi/fourier_neural_operator/data/planes.mat')
# csvwrite('/central/groups/tensorlab/khassibi/fourier_neural_operator/data/planes.csv', FileData.M)

# data = sns.load_dataset("/central/groups/tensorlab/khassibi/fourier_neural_operator/data/planes.csv")
# sns.jointplot(data=data, x="P_plane", y="V_plane")

# plt.show()

wandb.finish()