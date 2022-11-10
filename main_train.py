import numpy as np


import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch import optim




from miscellaneous import loss_plot


X = np.loadtxt('data_M10T50R02_constant.csv', dtype='float', delimiter=',')
y = np.loadtxt('targets_R00.csv', dtype='float', delimiter=',')

print(X.shape, X.dtype)
y = y.reshape(-1,1)
print(y.shape, y.dtype)



# Division into training and validation datasets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/2, random_state = 0, shuffle = False)

# Transformation to PyTorch tensors
X_train = torch.from_numpy(X_train)
X_test  = torch.from_numpy(X_test)
y_train = torch.from_numpy(y_train)
y_test  = torch.from_numpy(y_test)




print(X_train.shape, X_train.dtype)
print(y_train.shape, y_train.dtype)
print(X_test.shape, X_test.dtype)
print(y_test.shape, y_test.dtype)



# Training and validation datasets
ds_train = TensorDataset(X_train, y_train)
ds_test  = TensorDataset(X_test, y_test)

# DataLoaders
loader_train = DataLoader(ds_train, batch_size=100, shuffle=True)
loader_test  = DataLoader(ds_test,  batch_size=100, shuffle=False)

# Feedforward neural network
class Net(nn.Module):

    def __init__(self, n_in, n_mid, n_out):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_in,  n_mid)
        self.fc2 = nn.Linear(n_mid, n_mid)
        self.fc3 = nn.Linear(n_mid, n_mid)
        self.fc4 = nn.Linear(n_mid, n_mid)
        self.fc5 = nn.Linear(n_mid, n_mid)
        self.fc6 = nn.Linear(n_mid, n_out)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = torch.relu(self.fc5(x))
        output = self.fc6(x)
        return output    

# Initialization of random number sequence
torch.random.manual_seed(0)

# Network model
n_mid0 = 20     # No of nodes per hidden layer
model = Net(n_in = 15, n_mid = n_mid0, n_out = 1)  # R02

print(model)

params = list(model.parameters())


# Loss function
loss_fn = nn.MSELoss()

# Learning rate at first epoch
lr = 0.010

# Gradient descent method
optimizer = optim.Adam(model.parameters(), lr)

# No of epochs
n_epoch = 100
        
# Learning rate decay
def lr_scheduling(epoch):
    rate = 0.01
    return 1.0 - (1.0 - rate)*float(epoch)/float(n_epoch-1)


#########################
#   Training function   #
#########################

def train():
    model.train()
    for data, targets in loader_train:
        optimizer.zero_grad()
        outputs = model(data)
        loss = loss_fn(outputs, targets)
        loss.backward()
        optimizer.step()


###########################
#   Validation function   #
###########################

def test():
    model.eval()

    with torch.no_grad():
        outputs_train = model(X_train )
        outputs_test  = model(X_test )
        loss_train = loss_fn(outputs_train, y_train)
        loss_test  = loss_fn(outputs_test,  y_test )
        
    return loss_train, loss_test


#############################
#   Execution of training   #
#############################



# Loss function before training
loss_train, loss_test = test()



print("epoch {}".format(0))
print('loss_train: {:.5f}'.format(loss_train))
print('loss_test : {:.5f}'.format(loss_test))

vec_loss_train = [loss_train]
vec_loss_test  = [loss_test]

# Mini-batch training
scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_scheduling)

for epoch in range(n_epoch):
    train()
    loss_train, loss_test = test()
    if (epoch + 1)%10 == 0:
        print("epoch {}".format(epoch + 1))
        print('loss_train: {:.5f}'.format(loss_train))
        print('loss_test : {:.5f}'.format(loss_test))
       
    vec_loss_train.append(loss_train)
    vec_loss_test.append(loss_test)
    
    # Current learning rate
    for param in optimizer.param_groups:
        current_lr = param["lr"]

    # Update of learning rate
    scheduler.step()

# Plot of loss functions
ymin = 0.01
ymax = 10.0
loss_plot(vec_loss_train, vec_loss_test, 'Learning curves', ymin, ymax)

params = list(model.parameters())

print(params)

# File output of network parameters
torch.save(model.state_dict(), 'weight_M10T50R02_constant_E0')     # One member of a DNN ensemble


##############################
#   Outputs for validation   #
##############################

model.eval()

outputs = model(X_test)

print(X_test.shape, X_test.dtype)
print(X_test[0:10])
print(outputs.shape, outputs.dtype)
print(outputs[0:10])
print(y_test.shape, y_test.dtype)
print(y_test[0:10])

loss = loss_fn(outputs, y_test)

print('\n', 'loss function for validation : {:.5f}\n'.format(loss))