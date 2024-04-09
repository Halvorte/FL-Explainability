### Initialize model ###
import numpy as np
import importlib

fuzz_module = importlib.import_module("openfl-xai_workspaces.xai_tsk_frbs.src.model.fuzzySystem")
fuzzySystem_class = getattr(fuzz_module, "FuzzySystem")

#feature_names = ['number_of_elements', 'mean_atomic_mass', 'wtd_mean_atomic_mass', 'gmean_atomic_mass', 'wtd_gmean_atomic_mass', 'entropy_atomic_mass', 'wtd_entropy_atomic_mass', 'range_atomic_mass', 'wtd_range_atomic_mass', 'std_atomic_mass', 'wtd_std_atomic_mass', 'mean_fie', 'wtd_mean_fie', 'gmean_fie', 'wtd_gmean_fie', 'entropy_fie', 'wtd_entropy_fie', 'range_fie', 'wtd_range_fie', 'std_fie', 'wtd_std_fie', 'mean_atomic_radius', 'wtd_mean_atomic_radius', 'gmean_atomic_radius', 'wtd_gmean_atomic_radius', 'entropy_atomic_radius', 'wtd_entropy_atomic_radius', 'range_atomic_radius', 'wtd_range_atomic_radius', 'std_atomic_radius', 'wtd_std_atomic_radius', 'mean_Density', 'wtd_mean_Density', 'gmean_Density', 'wtd_gmean_Density', 'entropy_Density', 'wtd_entropy_Density', 'range_Density', 'wtd_range_Density', 'std_Density', 'wtd_std_Density', 'mean_ElectronAffinity', 'wtd_mean_ElectronAffinity', 'gmean_ElectronAffinity', 'wtd_gmean_ElectronAffinity', 'entropy_ElectronAffinity', 'wtd_entropy_ElectronAffinity', 'range_ElectronAffinity', 'wtd_range_ElectronAffinity', 'std_ElectronAffinity', 'wtd_std_ElectronAffinity', 'mean_FusionHeat', 'wtd_mean_FusionHeat', 'gmean_FusionHeat', 'wtd_gmean_FusionHeat', 'entropy_FusionHeat', 'wtd_entropy_FusionHeat', 'range_FusionHeat', 'wtd_range_FusionHeat', 'std_FusionHeat', 'wtd_std_FusionHeat', 'mean_ThermalConductivity', 'wtd_mean_ThermalConductivity', 'gmean_ThermalConductivity', 'wtd_gmean_ThermalConductivity', 'entropy_ThermalConductivity', 'wtd_entropy_ThermalConductivity', 'range_ThermalConductivity', 'wtd_range_ThermalConductivity', 'std_ThermalConductivity', 'wtd_std_ThermalConductivity', 'mean_Valence', 'wtd_mean_Valence', 'gmean_Valence', 'wtd_gmean_Valence', 'entropy_Valence', 'wtd_entropy_Valence', 'range_Valence', 'wtd_range_Valence', 'std_Valence', 'wtd_std_Valence']
#feature_names = ['lp', 'v', 'GTT', 'GTn', 'GGn', 'Ts', 'Tp', 'T48', 'T1', 'T2', 'P48', 'P1' ,'P2', 'Pexh', 'TIC', 'mf', 'GT1']
#feature_names = ['AT','V','AP','RH']
#feature_names = ['%IronFeed', '%SilicaFeed', 'StarchFlow', 'AminaFlow', 'OrePulpFlow', 'OrePulppH', 'OrePulpDensity', 'FlotationColumn01AirFlow', 'FlotationColumn02AirFlow', 'FlotationColumn03AirFlow', 'FlotationColumn04AirFlow', 'FlotationColumn05AirFlow', 'FlotationColumn06AirFlow', 'FlotationColumn07AirFlow', 'FlotationColumn01Level', 'FlotationColumn02Level', 'FlotationColumn03Level', 'FlotationColumn04Level', 'FlotationColumn05Level', 'FlotationColumn06Level', 'FlotationColumn07Level']
#feature_names = ['duration','protocol_type','service','flag','src_bytes','dst_bytes','land','wrong_fragment','urgent','hot','num_failed_logins','logged_in','num_compromised','root_shell','su_attempted','num_root','num_file_creations','num_shells','num_access_files','num_outbound_cmds','is_host_login','is_guest_login','count','srv_count','serror_rate','srv_serror_rate','rerror_rate','srv_rerror_rate','same_srv_rate','diff_srv_rate','srv_diff_host_rate','dst_host_count','dst_host_srv_count','dst_host_same_srv_rate','dst_host_diff_srv_rate','dst_host_same_src_port_rate','dst_host_srv_diff_host_rate','dst_host_serror_rate','dst_host_srv_serror_rate','dst_host_rerror_rate' ,'dst_host_srv_rerror_rate']
#feature_names = ['ps_ind_01','ps_ind_02_cat','ps_ind_03','ps_ind_04_cat','ps_ind_05_cat','ps_ind_06_bin','ps_ind_07_bin','ps_ind_08_bin','ps_ind_09_bin','ps_ind_10_bin','ps_ind_11_bin','ps_ind_12_bin','ps_ind_13_bin','ps_ind_14','ps_ind_15','ps_ind_16_bin','ps_ind_17_bin','ps_ind_18_bin','ps_reg_01','ps_reg_02','ps_reg_03','ps_car_01_cat','ps_car_02_cat','ps_car_03_cat','ps_car_04_cat','ps_car_05_cat','ps_car_06_cat','ps_car_07_cat','ps_car_08_cat','ps_car_09_cat','ps_car_10_cat','ps_car_11_cat','ps_car_11','ps_car_12','ps_car_13','ps_car_14','ps_car_15','ps_calc_01','ps_calc_02','ps_calc_03','ps_calc_04','ps_calc_05','ps_calc_06','ps_calc_07','ps_calc_08','ps_calc_09','ps_calc_10','ps_calc_11','ps_calc_12','ps_calc_13','ps_calc_14','ps_calc_15_bin','ps_calc_16_bin','ps_calc_17_bin','ps_calc_18_bin','ps_calc_19_bin','ps_calc_20_bin']
#feature_names = ["Time","V1","V2","V3","V4","V5","V6","V7","V8","V9","V10","V11","V12","V13","V14","V15","V16","V17","V18","V19","V20","V21","V22","V23","V24","V25","V26","V27","V28","Amount"]
#feature_names = ['tau1','tau2','tau3','tau4','p1','p2','p3','p4','g1','g2','g3','g4','stab']
feature_names = ['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age']
#feature_names = ['fixedacidity', 'volatileacidity', 'citricacid', 'residualsugar', 'chlorides', 'freesulfurdioxide', 'totalsulfurdioxide', 'density', 'pH', 'sulphates', 'alcohol']
#feature_names = ['ClimbRate', 'Sgz', 'P', 'Q', 'CurRoll', 'AbsRoll', 'DiffClb', 'DiffRollRate', 'DiffDiffClb', 'SaTime1', 'SaTime2', 'SaTime3', 'SaTime4', 'DiffSaTime1', 'DiffSaTime2', 'DiffSaTime3', 'DiffSaTime4', 'Sa', 'Goal']
#feature_names = ['climbRate', 'Sgz', 'p', 'q', 'curPitch', 'curRoll', 'absRoll', 'diffClb', 'diffRollRate', 'diffDiffClb', 'SeTime1', 'SeTime2', 'SeTime3', 'SeTime4', 'SeTime5', 'SeTime6', 'SeTime7', 'SeTime8', 'SeTime9', 'SeTime10', 'SeTime11', 'SeTime12', 'SeTime13', 'SeTime14', 'diffSeTime1', 'diffSeTime2', 'diffSeTime3', 'diffSeTime4', 'diffSeTime5', 'diffSeTime6', 'diffSeTime7', 'diffSeTime8', 'diffSeTime9', 'diffSeTime10', 'diffSeTime11', 'diffSeTime12', 'diffSeTime13', 'diffSeTime14', 'alpha', 'Se', 'goal']
#feature_names = ['Company', 'Time', 'FinancialDistress', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9', 'x10', 'x11', 'x12', 'x13', 'x14', 'x15', 'x16', 'x17', 'x18', 'x19', 'x20', 'x21', 'x22', 'x23', 'x24', 'x25', 'x26', 'x27', 'x28', 'x29', 'x30', 'x31', 'x32', 'x33', 'x34', 'x35', 'x36', 'x37', 'x38', 'x39', 'x40', 'x41', 'x42', 'x43', 'x44', 'x45', 'x46', 'x47', 'x48', 'x49', 'x50', 'x51', 'x52', 'x53', 'x54', 'x55', 'x56', 'x57', 'x58', 'x59', 'x60', 'x61', 'x62', 'x63', 'x64', 'x65', 'x66', 'x67', 'x68', 'x69', 'x70', 'x71', 'x72', 'x73', 'x74', 'x75', 'x76', 'x77', 'x78', 'x79', 'x80', 'x81', 'x82', 'x83']
#feature_names = ['Country', 'Year', 'Status', 'AdultMortality', 'infantdeaths', 'Alcohol', 'percentageexpenditure', 'HepatitisB', 'Measles', 'BMI', 'under-fivedeaths', 'Polio', 'Totalexpenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 'thinness1-19years', 'thinness5-9years', 'Incomecompositionofresources', 'Schooling']

antecedents = np.load("./global_models/tsk/TSK_global_model_rules_antec.npy")
consequents = np.load("./global_models/tsk/TSK_global_model_rules_conseq.npy")
weights = np.load("./global_models/tsk/TSK_global_model_weights.npy")

model = fuzzySystem_class(variable_names=feature_names,
            antecedents=antecedents,
            consequents=consequents,
            rule_weights=weights)


### Global interpretability ###
print("number of rules: " + str(model.get_number_of_rules()))
rules = model.__str__()
with open("./global_models/interpretable_rules.txt","w") as file:
  file.write(rules)


### Predict ####

X_test = np.load("./data/X_test.npy")
y_pred = model.predict(X_test)
y_true = np.load("./data/y_test.npy")

X_test = np.load("./data/X_test.npy")
y_true = np.load("./data/y_test.npy")
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
classifier = True

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

    # Pearson's correlation coefficient. score