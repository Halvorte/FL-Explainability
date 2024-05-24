import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import zipfile
#from sklearn.datasets import load_arff
from scipy.io import arff

def get_dataset_type():
    dataset_type = 'regression'
    return dataset_type

def is_classifier():
    return False

def get_dataset_name():
    name = 'ailerons'
    return name

def get_clean_dataset():
    # Read the data
    #arff_file = arff.loadarff('data/ailerons2.arff')
    arff_file = arff.loadarff('../data/aileronsCopy.dat') # Trying modified KEEL file
    df = pd.DataFrame(arff_file[0])
    #data, metadata = load_arff('data/aileronsCopy.dat')
    #df = pd.DataFrame(data, columns=metadata.feature_names)
    #df = pd.read_csv('data/ailerons.dat', delimiter=',')
    print(df.head())
    print(df)


    # Empty row/ missing data handling
    num_null = df.isnull().sum()
    print(f'nr of empty rows: {num_null}')
    missing_percentages = df.isnull().sum() / len(df) * 100
    print(f'Missing values percentages: {missing_percentages}')

    #print(df.info)
    #print(df.describe())

    # Normalize the train data
    #scaler = StandardScaler()
    scaler = MinMaxScaler()
    cols_to_normalize = [col for col in df.columns if col not in ['Goal']]
    df_normalized = df.copy()
    df_normalized[cols_to_normalize] = scaler.fit_transform(df_normalized[cols_to_normalize])

    #print(df_normalized)
    df_clean = df_normalized.dropna()
    return df_clean


def dl_dataloading(df_train, df_val, X_featrues, y_feature):
    # Split and save train data for DL model.Save training data without splitting
    DL_df_train = df_train.copy()
    train_X = DL_df_train.drop(y_feature, axis=1)
    train_y = DL_df_train[y_feature]
    np.save('../data/DL_X_train.npy', train_X.to_numpy())
    np.save('../data/DL_Y_train.npy', train_y.to_numpy())

    # Split val dataset into x and y, and save as npy files
    X_ = df_val.drop(y_feature, axis=1)
    Y_ = df_val[y_feature]
    print(Y_.to_numpy())
    # Save X as X_test.npy
    np.save("../data/X_test.npy", X_.to_numpy())
    # Save Y as y_test.npy
    np.save("../data/y_test.npy", Y_.to_numpy())
    print("Saved X_test and y_test as npy files successfully!")


def data_experiment_function(df_train, df_val, X_featrues, y_feature, experiment, clients, client_split=0.2):
    # Split train test
    #df_train, df_val = train_test_split(df_normalized, test_size=0.2, random_state=42)

    # Split train test and x and y
    #df_train, df_val = train_test_split(df, test_size=0.3, random_state=42)

    # Split and save train data for DL model.Save training data without splitting
    DL_df_train = df_train.copy()
    train_X = DL_df_train.drop(y_feature, axis=1)
    train_y = DL_df_train[y_feature]
    np.save('../data/DL_X_train.npy', train_X.to_numpy())
    np.save('../data/DL_Y_train.npy', train_y.to_numpy())

    # Split val dataset into x and y, and save as npy files
    # This is the data the global models are tested using
    X_ = df_val.drop(y_feature, axis=1)
    Y_ = df_val[y_feature]
    print(Y_.to_numpy())
    # Save X as X_test.npy
    np.save("../data/X_test.npy", X_.to_numpy())
    # Save Y as y_test.npy
    np.save("../data/y_test.npy", Y_.to_numpy())
    print("Saved X_test and y_test as npy files successfully!")

    # Split the n-datasets into x and y, and save in zip files
    clients = clients

    # split train into data for each runner
    train_datasets = np.array_split(df_train, clients)

    client = 0

    for data in train_datasets:
        X = data[X_featrues]
        Y = data[y_feature]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

        # Print the shapes of each dataset
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"Y_train shape: {y_train.shape}")
        print(f"Y_test shape: {y_test.shape}")

        # Save data as .npy files
        np.save("X_train.npy", X_train)
        np.save("X_test.npy", X_test)
        np.save("Y_train.npy", y_train)
        np.save("Y_test.npy", y_test)

        # Create a zip file and add the .npy files
        with zipfile.ZipFile(f"../data/col{client}_data.zip", "w") as zip_f:
            zip_f.write("X_train.npy")
            zip_f.write("X_test.npy")
            zip_f.write("y_train.npy")
            zip_f.write("y_test.npy")


        print(f"Data saved to col{client}_data.zip successfully!")
        client += 1

def get_data_features():
    X = ['ClimbRate', 'Sgz', 'P', 'Q', 'CurPitch', 'CurRoll', 'AbsRoll', 'DiffClb', 'DiffRollRate', 'DiffDiffClb', 'SeTime1', 'SeTime2', 'SeTime3', 'SeTime4', 'SeTime5', 'SeTime6', 'SeTime7', 'SeTime8', 'SeTime9', 'SeTime10', 'SeTime11', 'SeTime12', 'SeTime13', 'SeTime14', 'DiffSeTime1', 'DiffSeTime2', 'DiffSeTime3', 'DiffSeTime4', 'DiffSeTime5', 'DiffSeTime6', 'DiffSeTime7', 'DiffSeTime8', 'DiffSeTime9', 'DiffSeTime10', 'DiffSeTime11', 'DiffSeTime12', 'DiffSeTime13', 'DiffSeTime14', 'Alpha', 'Se']
    y = 'Goal'
    return X, y