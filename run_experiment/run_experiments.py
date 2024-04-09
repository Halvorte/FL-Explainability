import os
from sklearn.model_selection import KFold
'''
import dataloaders.Dataloader_ailerons
import dataloaders.Dataloader_combined_cycle_power_plant
import dataloaders.Dataloader_creditcard_fraud
import dataloaders.Dataloader_intrusion_detection_kdd
import dataloaders.Dataloader_life_expectancy
import dataloaders.Dataloader_maintenance_naval_propulsion
import dataloaders.Dataloader_mining_process
'''
import dataloaders.Dataloader_pima_indians_diabetes
'''
import dataloaders.Dataloader_smart_grid_stability
import dataloaders.Dataloader_superconductivity
import dataloaders.Dataloading_wine_quality
'''

# loop trough the files in dataloaders to
#files = [dataloaders.Dataloader_ailerons, dataloaders.Dataloader_combined_cycle_power_plant,
         #dataloaders.Dataloader_creditcard_fraud, dataloaders.Dataloader_intrusion_detection_kdd,
         #dataloaders.Dataloader_life_expectancy, dataloaders.Dataloader_maintenance_naval_propulsion,
         #dataloaders.Dataloader_mining_process, dataloaders.Dataloader_pima_indians_diabetes,
         #dataloaders.Dataloader_smart_grid_stability, dataloaders.Dataloader_superconductivity,
         #dataloaders.Dataloading_wine_quality]
test_file_ = dataloaders.Dataloader_pima_indians_diabetes

# get X and Y from the dataloaders
clean_data_df = test_file_.clean_data()

# get data features and target
X_feature, y_feature = test_file_.get_data_features()
#X = clean_data_df[[X_feature]]
#y = clean_data_df[y_feature]
# Create KFold
kf = KFold(n_splits=10, shuffle=True, random_state=42)

results = []

# loop trough the folds
for train_index, test_index in kf.split(clean_data_df):
    print(f'fold, train_set:{len(train_index)}, test_set: {len(test_index)}')

    # set up, train and save the model
    #X_train = X[train_index]
    #y_train = y[train_index]
    #X_test = X[test_index]
    #y_test = y[test_index]

    df_train = clean_data_df.iloc[train_index]
    df_test = clean_data_df.iloc[test_index]

    # save data in correct folders for project
    test_file_.data_experiment_function(df_train, df_test, X_feature, y_feature, experiment=True)
    #data_experiment_function(df_train, df_val, X_features, y_features, experiment=False)

    # get results from the model


