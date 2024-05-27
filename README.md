# Federated Machine Learning

This is the repository for the MSc thesis Federated Machine Learning. 

It contains the code for the experiments performed as described in the thesis.

This project repository started out as a fork of the [OpenFL-XAI](https://github.com/Unipisa/OpenFL-XAI) repository before 
The repository is a fork of the OpenFL-XAI framework with added code and datasets.


## How to run the experiments
Federated TSK-FRBS:
Run the run_experiments.py file. Choose datasets and set nr of clients to 5.

Central TSK-FRBS:
The central TSK-FRBS is the same as the federated but with 1 client.
Run the same run_experiments.py file, but change the nr of clients to 5, change the cols.yaml, data.yaml, and docker-compose.yaml file

Federated DL:
Select a dataset to get results from, and run the corresponding DL_Flower_<-dataset->.py file.

Central DL:
Select a dataset to get results from, and run the corresponding DL_<-dataset->.py file.


# Prerequisites

OpenFL-XAI requires:

- [Python 3](https://www.python.org/downloads/)
- [Docker](https://docs.docker.com/engine/install/)

DL models require:
- [PyTorch](https://pytorch.org)
- [Flower](https://flower.ai)

Other packages:

- NumPy >= 1.24.3
- SimpFul >= 2.11.0
- Scikit-learn >= 1.2.2
