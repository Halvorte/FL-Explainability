from sklearn.model_selection import KFold
import subprocess
import time
from results_files import get_results
import json

import dataloaders.Dataloader_ailerons
import dataloaders.Dataloader_combined_cycle_power_plant
import dataloaders.Dataloader_creditcard_fraud
import dataloaders.Dataloader_intrusion_detection_kdd
import dataloaders.Dataloader_life_expectancy
import dataloaders.Dataloader_maintenance_naval_propulsion
#import dataloaders.Dataloader_mining_process
import dataloaders.Dataloader_pima_indians_diabetes
import dataloaders.Dataloader_smart_grid_stability
import dataloaders.Dataloader_superconductivity
import dataloaders.Dataloading_wine_quality

# Helperfunction to run command
def run_command(command, working_dir):
    # run command and
    results = subprocess.run(command, shell=True, capture_output=True, cwd=working_dir)
    return results.stdout

# helper function to check for running containers
def check_containers_running(timeout=300):
    """Checks if any containers are exiting (finished) within a timeout."""
    start_time = time.time()
    while True:
        output = run_command("docker ps -a", working_dir)
        if "Exited (0)" not in output.decode():  # Check for exiting containers
            if time.time() - start_time > timeout:
                print(f"Timed out waiting for containers to finish (after {timeout} seconds).")
                return False
            time.sleep(5)  # Wait before next check
        else:
            return True  # Containers are finished

# loop trough the files in dataloaders to
#files = [dataloaders.Dataloader_ailerons, dataloaders.Dataloader_combined_cycle_power_plant, dataloaders.Dataloader_creditcard_fraud, dataloaders.Dataloader_intrusion_detection_kdd, dataloaders.Dataloader_life_expectancy, dataloaders.Dataloader_maintenance_naval_propulsion, dataloaders.Dataloader_pima_indians_diabetes, dataloaders.Dataloader_smart_grid_stability, dataloaders.Dataloader_superconductivity, dataloaders.Dataloading_wine_quality]
files = [dataloaders.Dataloader_creditcard_fraud, dataloaders.Dataloader_intrusion_detection_kdd, dataloaders.Dataloader_life_expectancy, dataloaders.Dataloader_maintenance_naval_propulsion, dataloaders.Dataloader_pima_indians_diabetes, dataloaders.Dataloader_smart_grid_stability, dataloaders.Dataloader_superconductivity, dataloaders.Dataloading_wine_quality]

    #dataloaders.Dataloader_ailerons, dataloaders.Dataloader_combined_cycle_power_plant,
         #dataloaders.Dataloader_creditcard_fraud, dataloaders.Dataloader_intrusion_detection_kdd,
         #dataloaders.Dataloader_life_expectancy, dataloaders.Dataloader_maintenance_naval_propulsion,
         #dataloaders.Dataloader_mining_process, dataloaders.Dataloader_pima_indians_diabetes,
         #dataloaders.Dataloader_smart_grid_stability, dataloaders.Dataloader_superconductivity,
         #dataloaders.Dataloading_wine_quality]
#files = [dataloaders.Dataloader_life_expectancy]
#files = [dataloaders.Dataloader_combined_cycle_power_plant]
#files = [dataloaders.Dataloader_life_expectancy]
#files = [dataloaders.Dataloader_pima_indians_diabetes, dataloaders.Dataloader_combined_cycle_power_plant]

#test_file_ = dataloaders.Dataloader_pima_indians_diabetes

nr_clients = 1

final_results = {}

for dataset_file in files:
    # get information about dataset if it is classifier or not
    dataset_type = dataset_file.is_classifier()

    dataset_name = dataset_file.get_dataset_name()

    # get X and Y from the dataloaders
    clean_data_df = dataset_file.get_clean_dataset()

    # get data features and target
    X_feature, y_feature = dataset_file.get_data_features()
    #X = clean_data_df[[X_feature]]
    #y = clean_data_df[y_feature]
    # Create KFold
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    results = []

    # loop through the folds
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
        dataset_file.data_experiment_function(df_train, df_test, X_feature, y_feature, experiment=True, clients=nr_clients)
        #data_experiment_function(df_train, df_val, X_features, y_features, experiment=False)

        working_dir = r'C:\Users\Halvor\Documents\Code\FL-Explainability'

        # get results from the model
        command1 = 'docker build -f Dockerfile.openfl_xai -t openfl_xai .'
        command2 = 'docker build -f Dockerfile.xai_aggregator -t openfl_xai/aggregator .'
        command3 = 'docker build -f Dockerfile.xai_collaborator -t openfl_xai/collaborator .'
        command4 = 'docker compose up -d'

        command5 = 'docker cp aggregator_xai:/current_workspace/TSK_global_model_rules_antec.npy ./global_models/tsk/TSK_global_model_rules_antec.npy'
        command6 = 'docker cp aggregator_xai:/current_workspace/TSK_global_model_rules_conseq.npy ./global_models/tsk/TSK_global_model_rules_conseq.npy'
        command7 = 'docker cp aggregator_xai:/current_workspace/TSK_global_model_weights.npy ./global_models/tsk/TSK_global_model_weights.npy'

        print(f'Starting commands')
        run_command(command1, working_dir)
        time.sleep(5)
        run_command(command2, working_dir)
        time.sleep(5)
        run_command(command3, working_dir)
        time.sleep(5)

        print(f'starting docker commands')
        run_command(command4, working_dir)
        if check_containers_running():
            time.sleep(15)
            print(f'Containers finished')
            run_command(command5, working_dir)
            run_command(command6, working_dir)
            run_command(command7, working_dir)

            results_ = get_results.get_model_results(X_feature, classifier=dataset_type)
            print(f'results from single run{results_}')
            results.append(results_)

        else:
            print(f'Failed to start containers or timed out')


    #avg_accuracy = sum(results) / len(results)
    #print(f'avg_accuracy: {avg_accuracy}')

    final_results[f'{dataset_name}'] = results
    #final_results.append(results)
    print(f'partly results: {results}')

print(final_results)
with open('experiment_results2.txt', 'w') as convert_file:
    convert_file.write(json.dumps(final_results))