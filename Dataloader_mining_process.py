import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
import zipfile

# Read the data
df = pd.read_csv('data/MiningProcess_Flotation_Plant_Database.csv', decimal=",")
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

# Convert columns with strings to numbers
#country_encoder = LabelEncoder()
#df['Country'] = country_encoder.fit_transform(df['Country'])

# Encode country column
#year_encoder = LabelEncoder()
#df['Year'] = year_encoder.fit_transform(df['Year'])

# Encode the status column
#status_encoder = LabelEncoder()
#df['Status'] = year_encoder.fit_transform(df['Status'])

#print(f'df_train head: {df.head()}')

# Drop the date column
df = df.drop(['date'], axis=1)
df = df.drop(['%IronConcentrate'], axis=1)
#print(df)

# Normalize the train data
#scaler = StandardScaler()
scaler = MinMaxScaler()
#cols = df_train.columns[df_train.columns != ['Country', 'Year', 'Status', 'Life expectancy']]
cols_to_normalize = [col for col in df.columns if col not in ['%IronFeed', '%SilicaFeed', '%SilicaConcentrate']]
df_normalized = df.copy()
df_normalized[cols_to_normalize] = scaler.fit_transform(df_normalized[cols_to_normalize])

#df_train[cols] = scaler.fit_transform(df_train[cols])
#df_train_normalized = pd.DataFrame(df_train_normalized, columns=['Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B', 'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure', 'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population', ' thinness  1-19 years', ' thinness 5-9 years', 'Income composition of resources', 'Schooling'])

print(df_normalized)

# drop emty rows.
# Can impute the values instead of removing in the future
df_clean = df_normalized.dropna()

# Split train test
df_train, df_val = train_test_split(df_clean, test_size=0.2, random_state=42)

# Split val dataset into x and y, and save as npy files
X_ = df_val.drop('%SilicaConcentrate', axis=1)
Y_ = df_val['%SilicaConcentrate']
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
    X = data[['%IronFeed', '%SilicaFeed', 'StarchFlow', 'AminaFlow', 'OrePulpFlow', 'OrePulppH', 'OrePulpDensity', 'FlotationColumn01AirFlow', 'FlotationColumn02AirFlow', 'FlotationColumn03AirFlow', 'FlotationColumn04AirFlow', 'FlotationColumn05AirFlow', 'FlotationColumn06AirFlow', 'FlotationColumn07AirFlow', 'FlotationColumn01Level', 'FlotationColumn02Level', 'FlotationColumn03Level', 'FlotationColumn04Level', 'FlotationColumn05Level', 'FlotationColumn06Level', 'FlotationColumn07Level']]
    Y = data['%SilicaConcentrate']

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