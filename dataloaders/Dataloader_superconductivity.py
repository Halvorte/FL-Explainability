import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import zipfile


def get_dataset_type():
    dataset_type = 'regression'
    return dataset_type

def is_classifier():
    return False

def get_dataset_name():
    name = 'superconductivity'
    return name


def get_clean_dataset():
    # Read the data
    df = pd.read_csv('../data/Superconductivity_data/train.csv')
    print(df)


    # Empty row/ missing data handling
    num_null = df.isnull().sum()
    print(f'nr of empty rows: {num_null}')
    missing_percentages = df.isnull().sum() / len(df) * 100
    print(f'Missing values percentages: {missing_percentages}')
    duplicated_data = df.duplicated().sum()
    print(f'Duplicated rows: {duplicated_data}')

    df = df.drop_duplicates()

    # fix spaces in column names
    df.rename(columns=lambda x: x.replace(' ', ''), inplace=True)


    # Normalize the train data
    #scaler = StandardScaler()
    scaler = MinMaxScaler()
    cols_to_normalize = [col for col in df.columns if col not in ['critical_temp']]
    df_normalized = df.copy()
    df_normalized[cols_to_normalize] = scaler.fit_transform(df_normalized[cols_to_normalize])

    print(df_normalized)

    # drop emty rows.
    # Can impute the values instead of removing in the future
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
    #df_train, df_val = train_test_split(df_clean, test_size=0.2, random_state=42)

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

    # split train into data for each runner
    clients = clients
    train_datasets = np.array_split(df_train, clients)
    print(train_datasets)

    # Split the 5 datasets into x and y, and save in zip files
    client = 0
    for data in train_datasets:
        #data = train_datasets[i]
        #X = data[['number_of_elements', 'mean_atomic_mass', 'wtd_mean_atomic_mass', 'gmean_atomic_mass', 'wtd_gmean_atomic_mass', 'entropy_atomic_mass', 'wtd_entropy_atomic_mass', 'range_atomic_mass', 'wtd_range_atomic_mass', 'std_atomic_mass', 'wtd_std_atomic_mass', 'mean_fie', 'wtd_mean_fie', 'gmean_fie', 'wtd_gmean_fie', 'entropy_fie', 'wtd_entropy_fie', 'range_fie', 'wtd_range_fie', 'std_fie', 'wtd_std_fie', 'mean_atomic_radius', 'wtd_mean_atomic_radius', 'gmean_atomic_radius', 'wtd_gmean_atomic_radius', 'entropy_atomic_radius', 'wtd_entropy_atomic_radius', 'range_atomic_radius', 'wtd_range_atomic_radius', 'std_atomic_radius', 'wtd_std_atomic_radius', 'mean_Density', 'wtd_mean_Density', 'gmean_Density', 'wtd_gmean_Density', 'entropy_Density', 'wtd_entropy_Density', 'range_Density', 'wtd_range_Density', 'std_Density', 'wtd_std_Density', 'mean_ElectronAffinity', 'wtd_mean_ElectronAffinity', 'gmean_ElectronAffinity', 'wtd_gmean_ElectronAffinity', 'entropy_ElectronAffinity', 'wtd_entropy_ElectronAffinity', 'range_ElectronAffinity', 'wtd_range_ElectronAffinity', 'std_ElectronAffinity', 'wtd_std_ElectronAffinity', 'mean_FusionHeat', 'wtd_mean_FusionHeat', 'gmean_FusionHeat', 'wtd_gmean_FusionHeat', 'entropy_FusionHeat', 'wtd_entropy_FusionHeat', 'range_FusionHeat', 'wtd_range_FusionHeat', 'std_FusionHeat', 'wtd_std_FusionHeat', 'mean_ThermalConductivity', 'wtd_mean_ThermalConductivity', 'gmean_ThermalConductivity', 'wtd_gmean_ThermalConductivity', 'entropy_ThermalConductivity', 'wtd_entropy_ThermalConductivity', 'range_ThermalConductivity', 'wtd_range_ThermalConductivity', 'std_ThermalConductivity', 'wtd_std_ThermalConductivity', 'mean_Valence', 'wtd_mean_Valence', 'gmean_Valence', 'wtd_gmean_Valence', 'entropy_Valence', 'wtd_entropy_Valence', 'range_Valence', 'wtd_range_Valence', 'std_Valence', 'wtd_std_Valence']]
        #Y = data['critical_temp']
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
    X = ['number_of_elements', 'mean_atomic_mass', 'wtd_mean_atomic_mass', 'gmean_atomic_mass', 'wtd_gmean_atomic_mass', 'entropy_atomic_mass', 'wtd_entropy_atomic_mass', 'range_atomic_mass', 'wtd_range_atomic_mass', 'std_atomic_mass', 'wtd_std_atomic_mass', 'mean_fie', 'wtd_mean_fie', 'gmean_fie', 'wtd_gmean_fie', 'entropy_fie', 'wtd_entropy_fie', 'range_fie', 'wtd_range_fie', 'std_fie', 'wtd_std_fie', 'mean_atomic_radius', 'wtd_mean_atomic_radius', 'gmean_atomic_radius', 'wtd_gmean_atomic_radius', 'entropy_atomic_radius', 'wtd_entropy_atomic_radius', 'range_atomic_radius', 'wtd_range_atomic_radius', 'std_atomic_radius', 'wtd_std_atomic_radius', 'mean_Density', 'wtd_mean_Density', 'gmean_Density', 'wtd_gmean_Density', 'entropy_Density', 'wtd_entropy_Density', 'range_Density', 'wtd_range_Density', 'std_Density', 'wtd_std_Density', 'mean_ElectronAffinity', 'wtd_mean_ElectronAffinity', 'gmean_ElectronAffinity', 'wtd_gmean_ElectronAffinity', 'entropy_ElectronAffinity', 'wtd_entropy_ElectronAffinity', 'range_ElectronAffinity', 'wtd_range_ElectronAffinity', 'std_ElectronAffinity', 'wtd_std_ElectronAffinity', 'mean_FusionHeat', 'wtd_mean_FusionHeat', 'gmean_FusionHeat', 'wtd_gmean_FusionHeat', 'entropy_FusionHeat', 'wtd_entropy_FusionHeat', 'range_FusionHeat', 'wtd_range_FusionHeat', 'std_FusionHeat', 'wtd_std_FusionHeat', 'mean_ThermalConductivity', 'wtd_mean_ThermalConductivity', 'gmean_ThermalConductivity', 'wtd_gmean_ThermalConductivity', 'entropy_ThermalConductivity', 'wtd_entropy_ThermalConductivity', 'range_ThermalConductivity', 'wtd_range_ThermalConductivity', 'std_ThermalConductivity', 'wtd_std_ThermalConductivity', 'mean_Valence', 'wtd_mean_Valence', 'gmean_Valence', 'wtd_gmean_Valence', 'entropy_Valence', 'wtd_entropy_Valence', 'range_Valence', 'wtd_range_Valence', 'std_Valence', 'wtd_std_Valence']
    y = 'critical_temp'
    return X, y