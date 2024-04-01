# Run a simulated cross-silo Federated Learning dataset on a Deep Learning model
import time

import numpy
# Imports
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
#import openfl.native as fx
#from openfl.federated import FederatedModel, FederatedDataSet
#from openfl.interface.interactive_api.experiment import TaskInterface, DataInterface, ModelInterface, FLExperiment
import copy
import os
import logging
import numpy as np
from tqdm import tqdm
from pprint import pprint
import torch

print('PyTorch', torch.__version__)

# Better CPU Utilization
os.environ['OMP_NUM_THREADS'] = str(int(os.cpu_count()))

# Get data after running the dataloader
train_x_data = numpy.load('data/DL_X_train.npy')
train_y_data = numpy.load('data/DL_Y_train.npy')
valid_x_data = numpy.load('data/X_test.npy')
valid_y_data = numpy.load('data/y_test.npy')

# Reshape the data to add a new dimension at the beginning (1, 7654)
train_y_data = train_y_data.reshape(-1, 1)
valid_y_data = valid_y_data.reshape(-1, 1)

num_features = valid_x_data.shape[1]
print(f'Num features: {num_features}')

train_x_data_tensor = torch.from_numpy(train_x_data).type(torch.Tensor)
train_y_data_tensor = torch.from_numpy(train_y_data).type(torch.Tensor)
valid_x_data_tensor = torch.from_numpy(valid_x_data).type(torch.Tensor)
valid_y_data_tensor = torch.from_numpy(valid_y_data).type(torch.Tensor)

# Model parameters
LEARNING_RATE = 0.0001
NUM_EPOCHS = 30
NUM_FEATURES = num_features
OUT_FEATURES = 1
BATCH_SIZE = 32
DEVICE = 'cpu'
DATASET = 'smart_grid_stability'
OPTIMIZER = 'Adam'

trainDataset = torch.utils.data.TensorDataset(train_x_data_tensor, train_y_data_tensor)
testDataset = torch.utils.data.TensorDataset(valid_x_data_tensor, valid_y_data_tensor)

# Create training and testing dataloader
train_dataloader = torch.utils.data.DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False)


class Net(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Net, self).__init__()
        self.lin1 = nn.Linear(in_features, 64)
        self.lin2 = nn.Linear(64, out_features)
        self.rel = nn.ReLU(inplace=True)
        self.drop = nn.Dropout(0.2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        x = self.lin1(x)
        x = self.rel(x)
        x = self.drop(x)

        x = self.lin2(x)
        x = self.sigmoid(x)
        return x

model = Net(NUM_FEATURES, OUT_FEATURES)
criterion = nn.MSELoss()      # Mean Square error loss function
#criterion = nn.CrossEntropyLoss()
#criterion = nn.L1Loss()
#criterion = nn.Sigmoid()
if OPTIMIZER == 'Adam':
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)    # Adam optimizer
else:
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
    correct += torch.sum(outputs.max(1)[1] == targets).item()

    ### Metrics ###
    targets_ = targets.detach().numpy()
    outputs_ = outputs.detach().numpy()
    #threshold on outputs
    for i in range(len(outputs_)):
        if outputs_[i] <= 0.5:
            outputs_[i]  = int(0)
        else:
            outputs_[i] = int(1)
    y_true = targets_.copy()
    y_pred = outputs_.copy()

    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    return {
        #'train_acc': np.round(correct / total, 3),
        'train_loss': np.round(np.mean(losses), 3),
        'accuracy': accuracy,
        'f1': f1,
        'precision': precision,
        'recall': recall,
    }


def validate(model, data, device, criterion):
    model.eval()
    model = model.to(device)

    losses = []
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in data:
            outputs = model(inputs.to(device))

            targets = targets.to(device)
            loss = criterion(outputs, targets)

            losses.append(loss.item())
            total += targets.shape[0]
            correct += (outputs.max(1)[1] == targets).sum().cpu().numpy()

            targets_ = targets.detach().numpy()
            outputs_ = outputs.detach().numpy()

            # threshold on outputs
            for i in range(len(outputs_)):
                if outputs_[i] <= 0.5:
                    outputs_[i] = int(0)
                else:
                    outputs_[i] = int(1)
            y_true = targets_
            y_pred = outputs_

            ### Metrics ###
            accuracy = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred)
            precision = precision_score(y_true, y_pred)
            recall = recall_score(y_true, y_pred)

        return {
            #'val_acc': np.round(correct / total, 3),
            'val_loss': np.round(np.mean(losses), 3),
            'accuracy': accuracy,
            'f1': f1,
            'precision':precision,
            'recall': recall,
        }


# Start!
time_start = time.time()
history = validate(model,
                   test_dataloader,
                   device=DEVICE,
                   criterion=criterion)
print('Before training: ', history)

val_losses = []
accuracies = []
f1_scores = []
precision_scores = []
recall_scores = []
for epoch in range(NUM_EPOCHS):
    train_history = train(model,
                          train_dataloader,
                          device=DEVICE,
                          optimizer=optimizer,
                          criterion=criterion)
    val_history = validate(model,
                           test_dataloader,
                           device=DEVICE,
                           criterion=criterion)
    print(f'Epoch {epoch}: {train_history} - {val_history} - time used:{time.time() - time_start}')
    val_losses.append(val_history['val_loss'])
    accuracies.append(val_history['accuracy'])
    f1_scores.append(val_history['f1'])
    precision_scores.append(val_history['precision'])
    recall_scores.append(val_history['recall'])


# get time used
time_now = time.time()
time_used = time_now - time_start

# Losses
nr = int(len(val_losses))
x_losses = list(range(nr))
y_losses = val_losses

plt.plot(x_losses, y_losses)
plt.title('Losses over rounds\n'
          f'{DATASET}')
plt.xlabel('Epochs')
plt.ylabel('MSE Loss')
#plt.ylim(0,25)
plt.savefig(f'images/losses_nonfl_{DATASET}.png')
plt.show()

nrr = int(len(accuracies))
x_acc = list(range(nrr))
y_acc = accuracies

plt.plot(x_acc, y_acc, color='red')
plt.title('Accuracy over rounds\n'
          f'{DATASET}')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
#plt.ylim(0.5,1.1)
plt.savefig(f'images/r2_accuracy_nonfl_{DATASET}.png')
plt.show()

print(f'batch size: {BATCH_SIZE}\n'
      f'epochs: {NUM_EPOCHS}\n'
      f'learning rate: {LEARNING_RATE}\n'
      f'accuracy: {accuracies[-1]}\n'
      f'f1: {f1_scores[-1]}\n'
      f'precision: {precision_scores[-1]}\n'
      f'recall: {recall_scores[-1]}\n'
      f'time used: {time_used}\n'
      f'optimizer: {OPTIMIZER}')
