import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
import subprocess
import time
#from results_files import get_results
import json
import numpy as np
import importlib

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

plot_information = []

# Helperfunction to run command
def run_command(command, working_dir):
    # run command and
    results = subprocess.run(command, shell=True, capture_output=True, cwd=working_dir)
    return results.stdout

# helper function to check for running containers
def check_containers_running(timeout=600):
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


def get_model_results(feature_names, classifier):
    #from openfl-xai_workspaces/xai_tsk_frbs/src/model/fuzzySystem.py import FuzzySystem

    #fuzz_module = fuzzySystem
    #fuzz_module = importlib.import_module("C:\Users\Halvor\Documents\Code\FL-Explainability\openfl-xai_workspaces\xai_tsk_frbs\src\model\fuzzySystem.py")
    fuzz_module = importlib.import_module("openfl-xai_workspaces.xai_tsk_frbs.src.model.fuzzySystem")
    fuzzySystem_class = getattr(fuzz_module, "FuzzySystem")

    antecedents = np.load("../global_models/tsk/TSK_global_model_rules_antec.npy")
    consequents = np.load("../global_models/tsk/TSK_global_model_rules_conseq.npy")
    weights = np.load("../global_models/tsk/TSK_global_model_weights.npy")

    print(f'antecedents shape: {antecedents.shape}')


    model = fuzzySystem_class(variable_names=feature_names,
                antecedents=antecedents,
                consequents=consequents,
                rule_weights=weights)


    ### Global interpretability ###
    print("number of rules: " + str(model.get_number_of_rules()))
    nr_of_rules = str(model.get_number_of_rules())
    rules = model.__str__()
    with open("../global_models/interpretable_rules.txt", "w") as file:
      file.write(rules)


    ### Predict ####

    #X_test = np.load("../data/X_test.npy")
    #y_pred = model.predict(X_test)
    #y_true = np.load("../data/y_test.npy")

    X_test = np.load("../data/X_test.npy")
    y_true = np.load("../data/y_test.npy")
    print(f'len feature_names: {len(feature_names)}')
    print(f'len data: {len(X_test)}')
    print(f'X_test shape: {X_test.shape}')
    y_pred_and_activated_rules_samples = model.predict_and_get_rule(X_test)
    y_pred = [tup[0] for tup in y_pred_and_activated_rules_samples]
    activated_rules = [tup[1] for tup in y_pred_and_activated_rules_samples]

    #Sometimes problem with this with some datasets.
    rule_adopted = model.get_rule_by_index(activated_rules[-1] + 0)
    print("last prediction: ")
    print("y_true: " + str(y_true[-1]))
    print("y_pred: "+str(y_pred[-1]))
    print("activated rule:")
    print(rule_adopted)

    # If it is a binary classification model, then make some changes to the output.
    #classifier = True

    classes = [0,1]
    if classifier:
        for i in range(len(y_pred)):
            #y_pred[i] = ( y_pred[i] - min(y_pred) ) / ( max(y_pred) - min(y_pred) )
            if y_pred[i] <= 0.5:
                y_pred[i]  = 0
            else:
                y_pred[i] = 1

        from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
        accuracy = accuracy_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)

        print(f'Classification accuracy: {accuracy}')
        print(f'F1 score: {f1}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')

        return [accuracy, f1, precision, recall, nr_of_rules]

    else:
        ### Metrics ###
        from sklearn.metrics import r2_score
        r2 = r2_score(y_true, y_pred)
        print(f'r2 accuracy: {r2}')

        # RMSE. Implement RMSE score.
        from sklearn.metrics import mean_squared_error
        mse = mean_squared_error(y_true, y_pred)
        print(f'mean squared error: {mse}')

        # mean absolute error
        from sklearn.metrics import mean_absolute_error
        mae = mean_absolute_error(y_true, y_pred)
        print(f'mean absolute error: {mae}')


        plot_information.append([y_true, y_pred])
        #y_ = y_true
        #y__ = y_pred
        #x_ = [i for i in range(len(y_))]
        #plt.plot(x_, y_, color='g')
        #plt.plot(x_, y__, color='r')
        #plt.show()

        # Pearson's correlation coefficient. score
        return [r2, mse, mae, nr_of_rules]

# loop trough the files in dataloaders to
#files = [dataloaders.Dataloader_ailerons, dataloaders.Dataloader_combined_cycle_power_plant, dataloaders.Dataloader_creditcard_fraud, dataloaders.Dataloader_intrusion_detection_kdd, dataloaders.Dataloader_life_expectancy, dataloaders.Dataloader_maintenance_naval_propulsion, dataloaders.Dataloader_pima_indians_diabetes, dataloaders.Dataloader_smart_grid_stability, dataloaders.Dataloader_superconductivity, dataloaders.Dataloading_wine_quality]

#files = [dataloaders.Dataloading_wine_quality]
files = [dataloaders.Dataloader_life_expectancy]
#files = [dataloaders.Dataloader_combined_cycle_power_plant]
#files = [dataloaders.Dataloader_life_expectancy]
#files = [dataloaders.Dataloader_pima_indians_diabetes]
#files = [dataloaders.Dataloader_pima_indians_diabetes, dataloaders.Dataloader_combined_cycle_power_plant]
#files = [dataloaders.Dataloader_superconductivity, dataloaders.Dataloading_wine_quality]

#test_file_ = dataloaders.Dataloader_pima_indians_diabetes

# Change between 1 and 5 depending on how many clients to run.
nr_clients = 5

final_results = {}
regression_plot_data = {}


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
    plot_information.clear()

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
        command0 = 'docker rm -f $(docker ps -a -q)'    # Delete all previous containers
        command1 = 'docker build -f Dockerfile.openfl_xai -t openfl_xai .'
        command2 = 'docker build -f Dockerfile.xai_aggregator -t openfl_xai/aggregator .'
        command3 = 'docker build -f Dockerfile.xai_collaborator -t openfl_xai/collaborator .'
        command4 = 'docker compose up -d'

        command5 = 'docker cp aggregator_xai:/current_workspace/TSK_global_model_rules_antec.npy ./global_models/tsk/TSK_global_model_rules_antec.npy'
        command6 = 'docker cp aggregator_xai:/current_workspace/TSK_global_model_rules_conseq.npy ./global_models/tsk/TSK_global_model_rules_conseq.npy'
        command7 = 'docker cp aggregator_xai:/current_workspace/TSK_global_model_weights.npy ./global_models/tsk/TSK_global_model_weights.npy'

        print(f'Starting commands')
        run_command(command0, working_dir)
        time.sleep(5)
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
            # Store the federated model
            run_command(command5, working_dir)
            time.sleep(5)
            run_command(command6, working_dir)
            time.sleep(5)
            run_command(command7, working_dir)
            time.sleep(5)

            results_ = get_model_results(X_feature, classifier=dataset_type)
            #results_ = get_results.get_model_results(X_feature, classifier=dataset_type)
            print(f'results from single run{results_}')
            results.append(results_)

        else:
            print(f'Failed to start containers or timed out')


    #avg_accuracy = sum(results) / len(results)
    #print(f'avg_accuracy: {avg_accuracy}')

    final_results[f'{dataset_name}'] = results
    regression_plot_data[f'{dataset_name}'] = plot_information
    #final_results.append(results)
    print(f'partly results: {results}')

print(f'{regression_plot_data}')

print(final_results)
with open('experiment_results_testing.txt', 'w') as convert_file:
    convert_file.write(json.dumps(final_results))

#json_string = json.dumps(regression_plot_data)
#with open('experiment_plots_1client.txt', 'w') as f:
#    f.write(json_string)

with open('experiment_plots_testing.txt', 'w') as f:
    for key, value_list in regression_plot_data.items():
        for inner_list in value_list:
            joined_values = ','.join(map(str, inner_list))  # Join values with comma
            f.write(f"{key}: {joined_values}\n")  # Write key-value pair with newline

print(plot_information)