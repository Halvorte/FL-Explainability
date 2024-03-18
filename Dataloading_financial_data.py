import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import zipfile

# Read the data
df = pd.read_csv('data/Financial Distress.csv')
print(df)

# Empty row/ missing data handling
num_null = df.isnull().sum()
print(f'nr of empty rows: {num_null}')
missing_percentages = df.isnull().sum() / len(df) * 100
print(f'Missing values percentages: {missing_percentages}')

# Remove country column
#df = df.drop(['Country'], axis=1)
#print(df)

# fix spaces in column names
df.rename(columns=lambda x: x.replace(' ', ''), inplace=True)


print(f'df_train head: {df.head()}')

# Normalize the train data
# Need to normalize using MinMaxScaler
#scaler = StandardScaler()
scaler = MinMaxScaler()
#cols = df_train.columns[df_train.columns != ['Company', 'Year', 'Status', 'Life expectancy']]
cols_to_normalize = [col for col in df.columns if col not in ['Company', 'Time', 'FinancialDistress']]
df_normalized = df.copy()
df_normalized[cols_to_normalize] = scaler.fit_transform(df_normalized[cols_to_normalize])

print(df_normalized)

# drop emty rows.
# Can impute the values instead of removing in the future
#df_clean = df_normalized.dropna()

# Split train test
df_train, df_val = train_test_split(df_normalized, test_size=0.2, random_state=42)

# Split val dataset into x and y, and save as npy files
X_ = df_val.drop('FinancialDistress', axis=1)
Y_ = df_val['FinancialDistress']
print(Y_.to_numpy())
# Save X as X_test.npy
np.save("data/X_test.npy", X_.to_numpy())
# Save Y as y_test.npy
np.save("data/y_test.npy", Y_.to_numpy())
print("Saved X_test and y_test as npy files successfully!")

# split train into data for each runner
train_datasets = np.array_split(df_train, 5)
print(train_datasets)

# Split the 5 datasets into x and y, and save in zip files
clients = 5
client = 0
for data in train_datasets:
    #data = train_datasets[i]
    X = data[['Company', 'Time', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31', 'x32', 'x33', 'x34', 'x35', 'x36', 'x37', 'x38', 'x39', 'x40', 'x41', 'x42', 'x43', 'x44', 'x45', 'x46', 'x47', 'x48', 'x49', 'x50', 'x51', 'x52', 'x53', 'x54', 'x55', 'x56', 'x57', 'x58', 'x59', 'x60', 'x61', 'x62', 'x63', 'x64', 'x65', 'x66', 'x67', 'x68', 'x69', 'x70', 'x71', 'x72', 'x73', 'x74', 'x75', 'x76', 'x77', 'x78', 'x79', 'x80', 'x81', 'x82', 'x83']]
    Y = data['FinancialDistress']

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
