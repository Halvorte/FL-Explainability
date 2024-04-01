import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import zipfile
from scipy.io import arff

# Read the Pima Indians Diabetes dataset
#arff_file = arff.loadarff('data/intrusion_detection_KDD/KDDTrain+_20Percent.arff')
#df = pd.DataFrame(arff_file[0])
df = pd.read_csv('data/intrusion_detection_KDD/KDDTrain+_20Percent.txt')
df = df[df.columns[:-1]]
print(df.head())

columns = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent'
            ,'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root'
            ,'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login'
            ,'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate'
            ,'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
            ,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate'
            ,'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate'
            ,'dst_host_srv_rerror_rate','class']
df.columns = columns

print(df)

# Empty row/ missing data handling
num_null = df.isnull().sum()
print(f'nr of empty rows: {num_null}')
missing_percentages = df.isnull().sum() / len(df) * 100
print(f'Missing values percentages: {missing_percentages}')

# fix spaces in column names
df.rename(columns=lambda x: x.replace(' ', ''), inplace=True)

# Convert columns with strings to numbers
protocol_encoder = LabelEncoder()
df['protocol_type'] = protocol_encoder.fit_transform(df['protocol_type'])

# Encode service column
service_encoder = LabelEncoder()
df['service'] = service_encoder.fit_transform(df['service'])

# Encode the flag column
flag_encoder = LabelEncoder()
df['flag'] = flag_encoder.fit_transform(df['flag'])

# Encode the class column
def custom_encoder(x):
  if x == 'normal':
    return 0
  else:
    return 1

# Apply the custom function to the 'class' column
df['class'] = df['class'].apply(custom_encoder)

print(f'df_train head: {df.head()}')

# Normalize the train data
#scaler = StandardScaler()
scaler = MinMaxScaler()
cols_to_normalize = [col for col in df.columns if col not in ['protocol_type' 'service', 'flag', 'land', 'logged_in', 'is_host_login', 'is_guest_login', 'class']]
df_normalized = df.copy()
df_normalized[cols_to_normalize] = scaler.fit_transform(df_normalized[cols_to_normalize])

print(df_normalized)

# drop emty rows.
# Can impute the values instead of removing in the future
df_clean = df_normalized.dropna()

# Split train test
df_train, df_val = train_test_split(df_clean, test_size=0.2, random_state=42)

# Split and save train data for DL model.Save training data without splitting
DL_df_train = df_train.copy()
train_X = DL_df_train.drop('class', axis=1)
train_y = DL_df_train['class']
np.save('data/DL_X_train.npy', train_X.to_numpy())
np.save('data/DL_Y_train.npy', train_y.to_numpy())

# Split val dataset into x and y, and save as npy files
X_ = df_val.drop('class', axis=1)
Y_ = df_val['class']
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
    X = data[['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent'
            ,'hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root'
            ,'num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login'
            ,'is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate'
            ,'same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count'
            ,'dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate'
            ,'dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate'
            ,'dst_host_srv_rerror_rate']]
    Y = data['class']

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