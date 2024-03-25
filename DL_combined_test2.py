# Run a simulated cross-silo Federated Learning dataset on a Deep Learning model
import numpy
# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms
import openfl.native as fx
from openfl.federated import FederatedModel, FederatedDataSet
from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface, FLExperiment
import copy
import os
import logging
import numpy as np
from tqdm import tqdm
from pprint import pprint

import torch
#import medmnist

print('PyTorch', torch.__version__)
#print('MedMNIST', medmnist.__version__)

# Better CPU Utilization
os.environ['OMP_NUM_THREADS'] = str(int(os.cpu_count()))

# Logging fix for Google Colab
log = logging.getLogger()
log.setLevel(logging.INFO)

# Switch to the tutorial directory within OpenFL tutorials
#tutorial_dir = os.path.abspath('openfl/openfl-tutorials/interactive_api/PyTorch_MedMNIST_2D')
#os.chdir(tutorial_dir)

# Set up workspace
#fx.init('DL_CCPowerPlant')

# Get data after running the dataloader
train_x_data = numpy.load('data/DL_X_train.npy')
train_y_data = numpy.load('data/DL_Y_train.npy')
valid_x_data = numpy.load('data/X_test.npy')
valid_y_data = numpy.load('data/y_test.npy')

# Reshape the data to add a new dimension at the beginning (1, 7654)
train_y_data = train_y_data.reshape(-1, 1)
valid_y_data = valid_y_data.reshape(-1, 1)

# Define helper function to convert NumPy arrays to PyTorch tensors
#def to_tensor(x, y):
  #return torch.Tensor(x), torch.Tensor(y)

# Convert every row to a tensor
#train_x_data_tensor = [torch.Tensor(row) for row in train_x_data]
#train_y_data_tensor = [torch.Tensor(row) for row in train_y_data]
#valid_x_data_tensor = [torch.Tensor(row) for row in valid_x_data]
#valid_y_data_tensor = [torch.Tensor(row) for row in valid_y_data]

#train_x_data_tensor = torch.tensor(train_x_data, dtype=torch.float32)
#train_y_data_tensor = torch.tensor(train_y_data, dtype=torch.float32)
#valid_x_data_tensor = torch.tensor(valid_x_data, dtype=torch.float32)
#valid_y_data_tensor = torch.tensor(valid_y_data, dtype=torch.float32)

train_x_data_tensor = torch.from_numpy(train_x_data).type(torch.Tensor)
train_y_data_tensor = torch.from_numpy(train_y_data).type(torch.Tensor)
valid_x_data_tensor = torch.from_numpy(valid_x_data).type(torch.Tensor)
valid_y_data_tensor = torch.from_numpy(valid_y_data).type(torch.Tensor)

# Convert data to Tensors
#train_data = to_tensor(train_x_data, train_y_data)
#test_data = to_tensor(valid_x_data, valid_y_data)


#train_data = train_x_data_tensor, train_y_data_tensor
#test_data = valid_x_data_tensor, valid_y_data_tensor
# Define batch size (adjust this value based on your needs)
batch_size = 16
num_workers = 6


trainDataset = torch.utils.data.TensorDataset(train_x_data_tensor, train_y_data_tensor)
#train_dataloader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

testDataset = torch.utils.data.TensorDataset(valid_x_data_tensor, valid_y_data_tensor)
#test_dataloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


# Create training and testing dataloader
train_dataloader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
#train_dataloader = torch.utils.data.DataLoader(list(zip(train_x_data_tensor, train_y_data_tensor)), batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False)
#test_dataloader = torch.utils.data.DataLoader(list(zip(valid_x_data_tensor, valid_y_data_tensor)), batch_size=batch_size, shuffle=False)
#dataloader = FederatedDataSet(train_x_data, train_y_data, valid_x_data, valid_y_data)


# Model parameters
LEARNING_RATE = 0.0001
NUM_CLIENTS = 5
NUM_EPOCHS = 20
ROUNDS_TO_TRAIN = 4
NUM_FEATURES = np.atleast_2d(train_x_data).shape[1]
print(f'Num features: {NUM_FEATURES}')
OUT_FEATURES = 1
BATCH_SIZE = 64
DEVICE = 'cpu'


class Net(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Net, self).__init__()
        self.lin1 = nn.Linear(in_features, 8)
        #self.lin2 = nn.Linear()
        self.lin2 = nn.Linear(8, out_features)
        self.rel = nn.ReLU()

    def forward(self, x):

        x = self.lin1(x)
        x = self.rel(x)
        x = self.lin2(x)
        return x

model = Net(NUM_FEATURES, OUT_FEATURES)
criterion = nn.MSELoss()      # Mean Square error loss function
#criterion = nn.Sigmoid()
#optimizer = optim.Adam(model.parameters(), LEARNING_RATE)    # Adam optimizer
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)     # Stochatic Gradient Descent
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
# Training/ Validation loops
def train(model, train_loader, optimizer, device, criterion):
    model.train()
    model = model.to(device)

    losses = []
    correct = 0
    total = 0
    #for inputs, targets in tqdm(train_loader, desc="train"):
    for inputs, targets in train_loader:

        optimizer.zero_grad()
        outputs = model(inputs.to(device))

        #targets = torch.squeeze(targets, 1).long().to(device)
        targets = targets.to(device)
        loss = criterion(outputs, targets)

        losses.append(loss.item())
        loss.backward()
        optimizer.step()

    total += targets.shape[0]
    #correct += torch.sum(outputs.max(1)[1] == targets).item()

    ### Metrics ###
    targets = targets.detach()
    outputs = outputs.detach()
    from sklearn.metrics import r2_score
    r2 = r2_score(targets, outputs)

    # RMSE. Implement RMSE score.
    from sklearn.metrics import mean_squared_error
    mse = mean_squared_error(targets, outputs)

    # mean absolute error
    from sklearn.metrics import mean_absolute_error
    mae = mean_absolute_error(targets, outputs)

    return {
        #'train_acc': np.round(correct / total, 3),
        'train_loss': np.round(np.mean(losses), 3),
        'r2': r2,
        'mse': mse,
    }, mse


def validate(model, data, device, criterion):
    model.eval()
    model = model.to(device)

    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        #for inputs, targets in tqdm(val_loader, desc="validate"):
        #for i in range(len(data[0].size(dim=0))):
        for inputs, targets in data:
            outputs = model(inputs.to(device))
            #outputs = model(data[0][i].to(device))

            #targets = torch.squeeze(targets, 1).long().to(device)
            targets = targets.to(device)
            loss = criterion(outputs, targets)

            losses.append(loss.item())
            total += targets.shape[0]
            correct += (outputs.max(1)[1] == targets).sum().cpu().numpy()

            ### Metrics ###
            from sklearn.metrics import r2_score
            r2 = r2_score(targets, outputs)

            # RMSE. Implement RMSE score.
            from sklearn.metrics import mean_squared_error
            mse = mean_squared_error(targets, outputs)

            # mean absolute error
            from sklearn.metrics import mean_absolute_error
            mae = mean_absolute_error(targets, outputs)

        return {
            #'val_acc': np.round(correct / total, 3),
            'val_loss': np.round(np.mean(losses), 3),
            'r2': r2,
            'mse': mse,
        }

# Train centralized model
#centralized_model = Net(NUM_FEATURES, OUT_FEATURES)
#optimizer = torch.optim.Adam(centralized_model.parameters(), lr=LEARNING_RATE)
#criterion = nn.CrossEntropyLoss()

# Start!
history = validate(model,
                   test_dataloader,
                   device=DEVICE,
                   criterion=criterion)
print('Before training: ', history)

mse_losses = []
for epoch in range(NUM_EPOCHS):
    train_history, mse_loss = train(model,
                          train_dataloader,
                          device=DEVICE,
                          optimizer=optimizer,
                          criterion=criterion)
    val_history = validate(model,
                           test_dataloader,
                           device=DEVICE,
                           criterion=criterion)
    print(f'Epoch {epoch}: {train_history} - {val_history}')
    mse_losses.append(mse_loss)

# Plot mse losses for training
import matplotlib.pyplot as plt
import numpy as np
# Losses
nr = int(len(mse_losses))
x_losses = list(range(nr))
y_losses = mse_losses

plt.plot(x_losses, y_losses)
plt.title('Losses over rounds')
plt.xlabel('epochs')
plt.ylabel('MSE Loss')
#plt.savefig('losses.png')
plt.show()

