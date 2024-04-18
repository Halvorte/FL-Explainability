import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import zipfile


def get_dataset_type():
    dataset_type = 'classification'
    return dataset_type

def is_classifier():
    return True

def get_dataset_name():
    name = 'pima_indians_diabetes'
    return name

def get_clean_dataset():
    # Read the Pima Indians Diabetes dataset
    df = pd.read_csv('../data/diabetes.csv')
    #print(df)

    # Empty row/ missing data handling
    num_null = df.isnull().sum()
    print(f'nr of empty rows: {num_null}')
    missing_percentages = df.isnull().sum() / len(df) * 100
    print(f'Missing values percentages: {missing_percentages}')


    # fix spaces in column names
    df.rename(columns=lambda x: x.replace(' ', ''), inplace=True)

    print(f'df_train head: {df.head()}')

    # Normalize the train data
    #scaler = StandardScaler()
    scaler = MinMaxScaler()
    cols_to_normalize = [col for col in df.columns if col not in ['Outcome']]
    df_normalized = df.copy()
    df_normalized[cols_to_normalize] = scaler.fit_transform(df_normalized[cols_to_normalize])

    print(df_normalized)

    # drop emty rows.
    # Can impute the values instead of removing in the future
    df_clean = df_normalized.dropna()
    return df_clean

def data_experiment_function(df_train, df_val, X_featrues, y_feature, experiment, clients, client_split=0.2):
    # Split train test
    #df_train, df_val = train_test_split(df_clean, test_size=client_split, random_state=42)

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
        #X = data[['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']]
        X = data[X_featrues]
        Y = data[y_feature]

        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=client_split, random_state=42)

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

    #return False

def get_data_features():
    X = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
    y = 'Outcome'
    return X, y

# If not using run experiment file and just want to run once
#clean_data = get_clean_dataset()
# Split train test
#df_train, df_val = train_test_split(clean_data, test_size=0.2, random_state=42)

#DL_df_train = df_train.copy()
##train_X = DL_df_train.drop('Outcome', axis=1)
#train_y = DL_df_train['Outcome']



#X_features, y_features = get_data_features()
#X = clean_data[X_features]
#y = clean_data[y_features]

#data_experiment_function(df_train, df_val, X_features, y_features, experiment=False)