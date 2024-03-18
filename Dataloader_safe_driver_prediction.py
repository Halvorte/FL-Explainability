import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import zipfile

# Read the Pima Indians Diabetes dataset
df = pd.read_csv('data/safe_driver_prediction/train.csv')
print(df)

# Empty row/ missing data handling
num_null = df.isnull().sum()
print(f'nr of empty rows: {num_null}')
missing_percentages = df.isnull().sum() / len(df) * 100
print(f'Missing values percentages: {missing_percentages}')

# fix spaces in column names
df.rename(columns=lambda x: x.replace(' ', ''), inplace=True)

# Drop id column
df = df.drop('id', axis=1)

print(f'df_train head: {df.head()}')

# Normalize the train data
#scaler = StandardScaler()
scaler = MinMaxScaler()
cols_to_normalize = [col for col in df.columns if col not in ['target']]
df_normalized = df.copy()
df_normalized[cols_to_normalize] = scaler.fit_transform(df_normalized[cols_to_normalize])

print(df_normalized)

# drop emty rows.
# Can impute the values instead of removing in the future
df_clean = df_normalized.dropna()

# Split train test
df_train, df_val = train_test_split(df_clean, test_size=0.2, random_state=42)

# Split val dataset into x and y, and save as npy files
X_ = df_val.drop('target', axis=1)
Y_ = df_val['target']
print(Y_.to_numpy())
# Save X as X_test.npy
np.save("data/X_test.npy", X_.to_numpy())
# Save Y as y_test.npy
np.save("data/y_test.npy", Y_.to_numpy())
print("Saved X_test and y_test as npy files successfully!")

# split train into data for each runner
clients = 5
train_datasets = np.array_split(df_train, clients)
print(train_datasets)

# Split the 5 datasets into x and y, and save in zip files
client = 0
for data in train_datasets:
    #data = train_datasets[i]
    X = data[['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin']]
    Y = data['target']

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
    with zipfile.ZipFile(f"data/col{client}_data.zip", "w") as zip_f:
        zip_f.write("X_train.npy")
        zip_f.write("X_test.npy")
        zip_f.write("y_train.npy")
        zip_f.write("y_test.npy")

    print(f"Data saved to col{client}_data.zip successfully!")
    client += 1