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

# Get data after running the dataloader
train_x_data = numpy.load('data/DL_X_train.npy')
train_y_data = numpy.load('data/DL_Y_train.npy')
valid_x_data = numpy.load('data/X_test.npy')
valid_y_data = numpy.load('data/y_test.npy')

# loop over every row and convert to list
#train_x_data = [row.tolist() for row in train_x_data]
#train_y_data = [row.tolist() for row in train_y_data]
#valid_x_data = [row.tolist() for row in valid_x_data]
#valid_y_data = [row.tolist() for row in valid_y_data]

#train_x_data = np.array(train_x_data)
#train_y_data = np.array(train_y_data)
#valid_x_data = np.array(valid_x_data)
#valid_y_data = np.array(valid_y_data)

#train_x_data = torch.from_numpy(train_x_data).float
#valid_x_data = torch.from_numpy(valid_x_data).float


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


federated_dataset = FederatedDataSet(train_x_data, train_y_data, valid_x_data, valid_y_data, batch_size=batch_size, num_classes=1)
#print(federated_dataset.get_feature_shape())

#trainDataset = torch.utils.data.TensorDataset(train_x_data_tensor, train_y_data_tensor)
#train_dataloader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)

#testDataset = torch.utils.data.TensorDataset(valid_x_data_tensor, valid_y_data_tensor)
#test_dataloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)


# Create training and testing dataloader
train_dataloader = torch.utils.data.DataLoader(trainDataset, batch_size=batch_size, shuffle=True)
#train_dataloader = torch.utils.data.DataLoader(list(zip(train_x_data_tensor, train_y_data_tensor)), batch_size=batch_size, shuffle=True)
test_dataloader = torch.utils.data.DataLoader(testDataset, batch_size=batch_size, shuffle=False)
#test_dataloader = torch.utils.data.DataLoader(list(zip(valid_x_data_tensor, valid_y_data_tensor)), batch_size=batch_size, shuffle=False)
#dataloader = FederatedDataSet(train_x_data, train_y_data, valid_x_data, valid_y_data)



##################
import numpy as np
from openfl.federated.data.loader import DataLoader

class CustomDataLoader(DataLoader):
    """
    Data Loader for in memory Numpy data.

    """

    def __init__(self, data_path, num_classes=None):
        """
        Initialize the training data from two numpy files named:
            - X_train.npy
            - y_train.npy

        Args:
            data_path: path to the numpy files folder.

            **kwargs: Additional arguments to pass to the function
        """
        super().__init__
        print("-------------------------------------------------------------------------------------------------------")
        X_train = np.load(data_path + '/DL_X_train.npy', mmap_mode=None, allow_pickle=False, fix_imports=True,
                          encoding='ASCII')
        y_train = np.load(data_path + '/DL_Y_train.npy', mmap_mode=None, allow_pickle=False, fix_imports=True,
                          encoding='ASCII')

        print(f'Training data loaded. Shape is {X_train.shape}')
        print(
            '--------------------------------------------------------------------------------------------------------')

        self.X_train = X_train
        self.y_train = y_train

        if num_classes is None:
            num_classes = np.unique(self.y_train).shape[0]
            print(f'Inferred {num_classes} classes from the provided labels...')

        self.num_classes = num_classes

    def get_train_data(self):
        """
        Get training data.

        Returns
        -------
            numpy.ndarray: training set
        """
        return self.X_train

    def get_train_labels(self):
        """
        Get training data labels.

        Returns
        -------
            numpy.ndarray: training set labels
        """
        return self.y_train

    def get_train_data_size(self):
        """
        Get total number of training samples.

        Returns:
            int: number of training samples
        """
        return len(self.X_train)

    def get_feature_shape(self):

        shape = self.X_train.shape[1]
        return shape

fl_data = CustomDataLoader(data_path='data', num_classes=1)

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

# Set up workspace
#fx.init('DL_CCPowerPlant')
fx.init()


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
#scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)
# Training/ Validation loops


#fl_model = FederatedModel(build_model=model, optimizer=optimizer, loss_fn=criterion, data_loader=federated_dataset)
fl_model = FederatedModel(build_model=model, optimizer=optimizer, loss_fn=criterion, data_loader=fl_data)

collaborator_models = fl_model.setup(num_collaborators=5)
collaborators = {'one': collaborator_models[0], 'two': collaborator_models[1], 'three': collaborator_models[2], 'four': collaborator_models[3], 'five': collaborator_models[4]}

print(fx.get_plan())

final_fl_model = fx.run_experiment(collaborators, override_config={'aggregator.settings.rounds_to_train': 30})

# save final model
final_fl_model.save_native('final_pytorch_model')

