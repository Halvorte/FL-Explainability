import torch
print(torch.__version__)

#!pip install -q flwr[simulation] torch torchvision matplotlib

# Flower file for the combined cycle power plant dataset

from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import time
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import CIFAR10
from torchvision.datasets import ImageFolder
from typing import Callable, Union
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
import flwr as fl
from flwr.common import Metrics
import dataloaders.Dataloading_wine_quality
from sklearn.model_selection import train_test_split


# Get the new dataset for Brain tumor MRI scans
def load_datasets():
    train_x_data = np.load('../data/DL_X_train.npy')
    train_y_data = np.load('../data/DL_Y_train.npy')
    valid_x_data = np.load('../data/X_test.npy')
    valid_y_data = np.load('../data/y_test.npy')

    train_y_data = train_y_data.reshape(-1, 1)
    valid_y_data = valid_y_data.reshape(-1, 1)

    num_features = valid_x_data.shape[1]

    valid_x_data_tensor = torch.from_numpy(valid_x_data).type(torch.Tensor)
    valid_y_data_tensor = torch.from_numpy(valid_y_data).type(torch.Tensor)

    # Split datasets into number of clients.
    # partition_size = int(len(train_data) / NUM_CLIENTS)
    partition_size = len(train_x_data) // NUM_CLIENTS
    remainders = len(train_x_data) % NUM_CLIENTS
    percentage = NUM_CLIENTS / 100
    lengths = [partition_size] * NUM_CLIENTS
    # lengths = [percentage] * NUM_CLIENTS
    num = remainders
    for i in range(len(lengths)):
        if num > 0:
            lengths[i] = lengths[i] + 1
        num -= 1
    generator = torch.Generator().manual_seed(42)
    print(len(train_x_data))
    print(lengths)
    print(f'remainders: {remainders}')

    testDataset = torch.utils.data.TensorDataset(valid_x_data_tensor, valid_y_data_tensor)

    # Create trainloaders and valloaders
    # Split train data into NUM_CLIENTS and then create a train and test set from that
    trainloaders = []
    valloaders = []
    nr = 0
    for i in lengths:
        x = train_x_data[nr:(nr + i), :]
        y = train_y_data[nr:(nr + i)]

        train_x_data_partial, test_x_data_partial, train_y_data_partial, test_y_data_partial = train_test_split(x, y, test_size=0.2, random_state=42)

        train_x_data_tensor = torch.from_numpy(train_x_data_partial).type(torch.Tensor)
        train_y_data_tensor = torch.from_numpy(train_y_data_partial).type(torch.Tensor)
        test_x_data_tensor = torch.from_numpy(test_x_data_partial).type(torch.Tensor)
        test_y_data_tensor = torch.from_numpy(test_y_data_partial).type(torch.Tensor)

        train_dataset = torch.utils.data.TensorDataset(train_x_data_tensor, train_y_data_tensor)
        test_dataset = torch.utils.data.TensorDataset(test_x_data_tensor, test_y_data_tensor)
        train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
        trainloaders.append(train_dataloader)
        valloaders.append(test_dataloader)

        nr = nr + i


    testloader = DataLoader(testDataset, batch_size=BATCH_SIZE)

    return trainloaders, valloaders, testloader, num_features


class Net(nn.Module):
    def __init__(self, in_features: int, out_features: int) -> None:
        super(Net, self).__init__()
        self.lin1 = nn.Linear(in_features, 128)
        self.lin2 = nn.Linear(128, 64)
        self.lin3 = nn.Linear(64, out_features)
        self.rel = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        #self.batch_norm1 = nn.BatchNorm1d(128)
        #self.batch_norm2 = nn.BatchNorm1d(64)

    def forward(self, x):
        x = self.lin1(x)
        #x = self.batch_norm1(x)
        x = self.rel(x)
        x = self.dropout(x)

        x = self.lin2(x)
        #x = self.batch_norm2(x)
        x = self.rel(x)
        x = self.dropout(x)

        x = self.lin3(x)
        return x



def train(net, trainloader, epochs: int, verbose=False):
    """Train the network on the training set."""
    #print('train function running!')
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()
    if OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), LEARNING_RATE)
    else:
        optimizer = torch.optim.SGD(net.parameters(), lr=LEARNING_RATE)
    net.train()
    for epoch in range(epochs):
        r2_scores, total, epoch_loss = 0, 0, 0.0
        for inputs, targets in trainloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            #print(f'data shape: {images.shape}')
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            # Metrics
            epoch_loss += loss
            #print(f'targets and outputs: {targets}, {outputs}')
            targets = targets.detach().numpy()
            outputs = outputs.detach().numpy()
            r2 = r2_score(targets, outputs)
            r2_scores += r2
            total += 1
        epoch_loss /= len(trainloader.dataset)
        avg_r2 = r2_scores / total
        #r2_scores = r2
        if verbose:
            print(f"Epoch {epoch+1}: train loss {epoch_loss}, avg-epoch-r2 {avg_r2}")


def test(net, testloader):
    """Evaluate the network on the entire test set."""
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.MSELoss()
    r2_scores, mse_scores, mae_scores, total, loss = 0, 0, 0, 0, 0.0
    net.eval()
    with torch.no_grad():
        for inputs, targets in testloader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = net(inputs)
            loss += criterion(outputs, targets).item()
            _, predicted = torch.max(outputs.data, 1)
            total += 1
            predicted_.append(predicted)    # Add predicted to list so it is possible to create confusion matrix
            true_.append(targets)
            targets = targets.detach().numpy()
            outputs = outputs.detach().numpy()
            r2 = r2_score(targets, outputs)
            mse = mean_squared_error(targets, outputs)
            mae = mean_absolute_error(targets, outputs)
            r2_scores += r2
            mse_scores += mse
            mae_scores += mae
    loss /= len(testloader.dataset)
    r2_accuracy = r2_scores / total
    mse_accuracy = mse_scores / total
    mae_accuracy = mae / total
    return loss, r2_accuracy, mse_accuracy, mae_accuracy


def get_parameters(net) -> List[np.ndarray]:
    #print('get_parameters. state_dict')
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_parameters(net, parameters: List[np.ndarray]):
    #print('set_parameters. state_dict')
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)

class FlowerClient(fl.client.NumPyClient):
    def __init__(self, net, trainloader, valloader):
        self.net = net
        self.trainloader = trainloader
        self.valloader = valloader

    def get_parameters(self, config):
        return get_parameters(self.net)

    def fit(self, parameters, config):
        set_parameters(self.net, parameters)
        train(self.net, self.trainloader, epochs=CLIENT_EPOCHS)
        #print('fittin!')
        return get_parameters(self.net), len(self.trainloader), {}

    def evaluate(self, parameters, config):
        set_parameters(self.net, parameters)
        loss, r2_accuracy, mse_accuracy, mae_accuracy = test(self.net, self.valloader)
        return float(loss), len(self.valloader), {"accuracy": float(r2_accuracy)}


def client_fn(cid: str) -> FlowerClient:
    """Create a Flower client representing a single organization."""

    # Load model
    net = Net(NUM_FEATURES, OUT_FEATURES).to(DEVICE)

    # Load data (CIFAR-10)
    # Note: each client gets a different trainloader/valloader, so each client
    # will train and evaluate on their own unique data
    trainloader = trainloaders[int(cid)]
    valloader = valloaders[int(cid)]
    # print(f'client_fn, train and valloader nr {int(cid)}')

    # Create a  single Flower client representing a single organization
    return FlowerClient(net, trainloader, valloader).to_client()


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    # r2 * num_examples
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# The `evaluate` function will be by Flower called after every round
# Centralized evaluation of the aggregated model

def evaluate(
    server_round: int,
    parameters: fl.common.NDArrays,
    config: Dict[str, fl.common.Scalar],
) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
    net = Net(NUM_FEATURES, OUT_FEATURES).to(DEVICE)
    valloader = valloaders[0]
    set_parameters(net, parameters)  # Update model with the latest parameters
    loss, r2_accuracy, mse_accuracy, mae_accuracy = test(net, valloader)
    print(f"Server-side evaluation loss {loss}, r2-accuracy {r2_accuracy}, mse {mse_accuracy}, mae {mae_accuracy}, round {server_round}")
    accuracies.append(r2_accuracy)
    losses.append(loss)
    mae_losses.append(mae_accuracy)
    time_now = time.time()
    time_sofar = time_now - TIME
    time_used.append(time_sofar)
    print(f'Time used so far: {time_sofar}')
    return loss, {"accuracy": r2_accuracy}


# Function making it possible to sent parameters to the clients
def get_on_fit_config_fn() -> Callable[[int], Dict[str, str]]:
    """Return a function which returns training configurations."""

    def fit_config(server_round: int) -> Dict[str, str]:
        """Return a configuration with static batch size and (local) epochs."""
        config = {
            "learning_rate": str(LEARNING_RATE),
            "batch_size": str(BATCH_SIZE),
        }
        return config

    return fit_config



dataset_name = dataloaders.Dataloading_wine_quality.get_dataset_name()
# Run dataloader file to make data available.
clean_data_df = dataloaders.Dataloading_wine_quality.get_clean_dataset()
X_feature, y_feature = dataloaders.Dataloading_wine_quality.get_data_features()

kf = KFold(n_splits=10, shuffle=True, random_state=42)

results = []

# loop through the folds
for train_index, test_index in kf.split(clean_data_df):
    print(f'fold, train_set:{len(train_index)}, test_set: {len(test_index)}')

    df_train = clean_data_df.iloc[train_index]
    df_test = clean_data_df.iloc[test_index]

    # save data in correct folders for project
    dataloaders.Dataloading_wine_quality.dl_dataloading(df_train, df_test, X_feature, y_feature)


    # Run dataloader file to make data available.
    #clean_data_df = dataloaders.Dataloading_wine_quality.get_clean_dataset()
    #X_feature, y_feature = dataloaders.Dataloading_wine_quality.get_data_features()

    # split train and val
    #df_train, df_val = train_test_split(clean_data_df, test_size=0.2, random_state=42)

    #dataloaders.Dataloading_wine_quality.dl_dataloading(df_train, df_val, X_feature, y_feature)

    torch.cuda.empty_cache()

    if torch.cuda.is_available():
        #DEVICE = torch.device("cuda")  # Try "cuda" to train on GPU
        DEVICE = torch.device("cpu")
        print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")
    else:
        DEVICE = torch.device("cpu")
        print(f"Training on {DEVICE} using PyTorch {torch.__version__} and Flower {fl.__version__}")

    DATASET = 'red_wine_quality'
    NUM_CLIENTS = 5
    BATCH_SIZE = 32
    CLIENT_EPOCHS = 150
    NR_ROUNDS = 10   # nr of rounds the federated learning should do
    LEARNING_RATE = 0.0001
    OPTIMIZER = 'Adam'

    TIME = time.time()
    TIME_ = time.time()

    accuracies = []
    losses = []
    mae_losses = []
    time_used = []

    predicted_ = []
    true_ = []


    trainloaders, valloaders, testloader, num_features = load_datasets()




    NUM_FEATURES = num_features
    print(f'Num features: {NUM_FEATURES}')
    OUT_FEATURES = 1
    model = Net(NUM_FEATURES, OUT_FEATURES)
    criterion = nn.MSELoss()      # Mean Square error loss function
    #criterion = nn.Sigmoid()
    if OPTIMIZER == 'Adam':
        optimizer = torch.optim.Adam(model.parameters(), LEARNING_RATE)    # Adam optimizer
    else:
        optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)     # Stochatic Gradient Descent
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, verbose=True)



    # Create FedAvg strategy
    strategy = fl.server.strategy.FedAvg(
        # strategy = fl.server.strategy.FedAdagrad(
        # Fraction_fit means how many of the clients participate every round. 1.0 = 100%, 0.5 = 50%
        fraction_fit=1.0,
        fraction_evaluate=0.5,  # percentage of clients randomly selected for evaluation
        min_fit_clients=NUM_CLIENTS,  # could be lower than num_clients
        # min_fit_clients=5,
        min_evaluate_clients=NUM_CLIENTS,
        # min_evaluate_clients=10,
        min_available_clients=NUM_CLIENTS,
        # min_available_clients=10,   # Wait until 10 clients are available should be NUM_CLIENTS
        evaluate_metrics_aggregation_fn=weighted_average,  # <-- pass the metric aggregation function
        evaluate_fn=evaluate,
        on_fit_config_fn=get_on_fit_config_fn(),
        # initial_parameters=fl.common.ndarrays_to_parameters(get_parameters(Net())), # used for FedAdagrad
    )

    # Specify client resources if you need GPU (defaults to 1 CPU and 0 GPU)
    client_resources = None
    if DEVICE.type == "cuda":
        client_resources = {"num_gpus": 1}
        print('cuda?')

    # print('Yo got here!')

    # Start simulation
    fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=NUM_CLIENTS,
        config=fl.server.ServerConfig(num_rounds=NR_ROUNDS),  # num_rounds = 10
        strategy=strategy,
        client_resources=client_resources,
    )

    import matplotlib.pyplot as plt
    import numpy as np


    # Losses
    nr = int(len(losses))
    x_losses = list(range(nr))
    y_losses = losses

    plt.plot(x_losses, y_losses)
    plt.title('Losses over rounds\n'
              f'{DATASET}')
    plt.xlabel('Rounds of federated learning')
    plt.ylabel('Loss')
    plt.ylim(0,0.04)
    plt.savefig(f'../images/losses_fl_{DATASET}.png')
    plt.show()

    # Accuracy
    nrr = int(len(accuracies))
    x_acc = list(range(nrr))
    y_acc = accuracies

    plt.plot(x_acc, y_acc, color='red')
    plt.title('R2-Accuracy over rounds\n'
              f'{DATASET}')
    plt.xlabel('Rounds of federated learning')
    plt.ylabel('Accuracy')
    plt.ylim(0,1.1)
    plt.savefig(f'../images/r2_accuracy_fl_{DATASET}.png')
    plt.show()


    print(f'nr of clients: {NUM_CLIENTS}\n'
          f'batch size: {BATCH_SIZE}\n'
          f'client epochs: {CLIENT_EPOCHS}\n'
          f'nr of federated rounds: {NR_ROUNDS}\n'
          f'learning rate: {LEARNING_RATE}\n'
          f'optimizer: {OPTIMIZER}\n'
          f'R2: {accuracies[-1]}\n'
          f'MSE: {losses[-1]}\n'
          f'MAE: {mae_losses[-1]}\n'
          f'Time_used: {time_used[-1]}')

    # Add results
    results_dict = {'dataset': dataset_name, 'R2': accuracies[-1], 'MSE': losses[-1], 'MAE': mae_losses[-1]}
    results.append(results_dict)

print(f'results:{results}')

# calculate mean results and 90% confidence interval
# Extract R2 values from each dictionary
r2_values = [d['R2'] for d in results]
mean_r2 = np.mean(r2_values)
std_r2 = np.std(r2_values)
z_score = 1.645  # Define z-score for 90% confidence interval (around 1.645)
margin_of_error = z_score * std_r2  # Calculate the margin of error
# confidence_interval_percentage = margin_of_error

mse_values = [d['MSE'] for d in results]
mean_mse = np.mean(mse_values)
std_mse = np.std(mse_values)
margin_of_error_mse = z_score * std_mse  # Calculate the margin of error

mae_values = [d['MAE'] for d in results]
mean_mae = np.mean(mae_values)
std_mae = np.std(mae_values)
margin_of_error_mae = z_score * std_mae  # Calculate the margin of error

print(f"mean R2: {mean_r2}, margin: {margin_of_error}")
print(f"mean MSE: {mean_mse}, margin: {margin_of_error_mse}")
print(f"mean MAE: {mean_mae}, margin: {margin_of_error_mae}")