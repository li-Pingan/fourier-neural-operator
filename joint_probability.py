import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import torch
import seaborn as sns


# mat = scipy.io.loadmat('//central/groups/tensorlab/khassibi/fourier_neural_operator/data/planes.mat')

# print(mat['P_plane'])

TRAIN_PATH = '/central/groups/tensorlab/khassibi/fourier_neural_operator/data/planes.mat'
TEST_PATH = '/central/groups/tensorlab/khassibi/fourier_neural_operator/data/planes.mat'

ntrain = 3000
ntest = 1000

r = 1
s1 = 768//r
s2 = 288//r

idx = torch.arange(ntrain + ntest)
training_idx = idx[:ntrain]
testing_idx = idx[-ntest:]

reader = MatReader(TRAIN_PATH)
x_train = reader.read_field('P_plane').permute(2,0,1)[training_idx][:,::r,::r][:,:s1,:s2]
y_train = reader.read_field('V_plane').permute(2,0,1)[training_idx][:,::r,::r][:,:s1,:s2]

x_normalizer = UnitGaussianNormalizer(x_train)
x_train = x_normalizer.encode(x_train)
x_test = x_normalizer.encode(x_test)

y_normalizer = UnitGaussianNormalizer(y_train)
y_train = y_normalizer.encode(y_train)

x_train = x_train.reshape(ntrain,s1,s2,1)
x_test = x_test.reshape(ntest,s1,s2,1)

mean_train_pressure = []
for i in range(len(x_train)):
    sum = 0
    for j in range(len(x_train[i])):
        for k in range(len(x_train[i][j])):
            sum += x_train[i][j][k]
    sum /= len(x_train[i]) * len(x_train[i][j])
    mean_train_pressure.append(sum)

mean_train_velocity = []
for i in range(len(y_train)):
    sum = 0
    for j in range(len(y_train[i])):
        for k in range(len(y_train[i][j])):
            sum += x_train[i][j][k]
    sum /= len(y_train[i]) * len(y_train[i][j])
    mean_train_velocity.append(sum)

mean_test_pressure = []
for i in range(len(x_test)):
    sum = 0
    for j in range(len(x_test[i])):
        for k in range(len(x_test[i][j])):
            sum += x_train[i][j][k]
    sum /= len(x_test[i]) * len(x_test[i][j])
    mean_test_pressure.append(sum)

mean_test_velocity = []
for i in range(len(y_test)):
    sum = 0
    for j in range(len(y_test[i])):
        for k in range(len(y_test[i][j])):
            sum += x_train[i][j][k]
    sum /= len(y_test[i]) * len(y_test[i][j])
    mean_test_velocity.append(sum)

  
# draw jointplot with
# kde kind
sns.jointplot(x = "id", y = "pulse",
              kind = "kde", data = data)
# Show the plot
plt.show()
  
# FileData = load('/central/groups/tensorlab/khassibi/fourier_neural_operator/data/planes.mat')
# csvwrite('/central/groups/tensorlab/khassibi/fourier_neural_operator/data/planes.csv', FileData.M)

# data = sns.load_dataset("/central/groups/tensorlab/khassibi/fourier_neural_operator/data/planes.csv")
# sns.jointplot(data=data, x="P_plane", y="V_plane")

# plt.show()

