
from sklearn.preprocessing import LabelEncoder
from MultilayerPerceptron.MLP_1_layer import *
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR

from neuroHarmonize import harmonizationLearn

from Utils import *
import pickle

# save dir
config_parser = configparser.ConfigParser(allow_no_value=True)
bindir = os.path.abspath(os.path.dirname(__file__))
config_parser.read(bindir + "/cfg.cnf")

# define a directory to save results
save_dir = config_parser.get("RESULTS", "Results")

# Load your data must follow the format of data_mockup.csv in /DataCreation
# all_data = load_your_data()

all_data, edades_todos = cargo_datos_todos()
edades_todos = edades_todos.reset_index(drop=True)
all_data.reset_index(inplace=True, drop=True)

# Añado la edad al dataframe para facilitarme el proceso
all_data['Age'] = edades_todos

# Save individuals characteristics
scan_list_all, etiv_all, ages_all = all_data['Escaner'].values, all_data['eTIV'].values, all_data['Age']
bo_todos, sex_all, IDs = all_data['Bo'].values, all_data['M/F'].values, all_data['ID'].values
BDs, Patologia = all_data['BD'].values, all_data['Patologia'].values
all_data = all_data.drop(['ID', 'Bo', 'M/F', 'Escaner', 'BD', 'Patologia', 'Age'], axis=1)

# Low variance features; might obstruct the harmonization procedure. Erase them if you are going to perform harmonization
columns_to_drop = [96, 97, 98, 99, 100, 101, 180, 181, 182, 183, 184, 185, 192, 193, 194, 195, 196, 197, 204, 205, 206,
                   207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226,
                   227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239]
features_to_drop = all_data.columns[columns_to_drop]
all_data = all_data.drop(features_to_drop, axis=1)
features_names = all_data.columns.tolist()

# Label encoder scaner and sex
data_scan_num = pd.DataFrame(LabelEncoder().fit_transform(scan_list_all))
data_sex_num = pd.DataFrame(LabelEncoder().fit_transform(sex_all))

# build covariate matrix nad harmonize the data 
d = {'SITE': data_scan_num.values.tolist(), 'SEX': np.squeeze(data_sex_num.values).tolist(), 'ETIV': etiv_all.tolist(), 'AGE':ages_all.values.tolist()}
my_model, data = harmonizationLearn(all_data.values, pd.DataFrame(data=d))

data = pd.DataFrame(data, columns=features_names)
data = data.assign(Age=ages_all.values, Escaner=scan_list_all, Bo=bo_todos,
                   **{'M/F': sex_all, 'eTIV': etiv_all, 'ID': IDs, 'BD': BDs, 'Patologia': Patologia})

# randomize your data order, reset index
data = data.sample(frac=1, random_state=42).reset_index(drop=True)

# Create lists and dataframes to save results
features, features_tag, MAE_val, MAE_test = [], [], [], []
prediction_SVR_saved_test, prediction_perceptron_saved_test, prediction_RandomForest_saved_test = [], [], []
prediction_SVR_saved_val, prediction_perceptron_saved_val, prediction_RandomForest_saved_val = [], [], []
results_dataframe_SVR_test, results_dataframe_SVR_test, results_dataframe_RF_test,\
    results_dataframe_SVR_val, results_dataframe_perceptron_val, results_dataframe_RF_val = \
    pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

# 10 fold cross validation, 20, 30, 40 features
for j in [20, 30, 40]:
    for i in range(0, 10, 1):

        # split the data for the 10 fold cross validation
        data_train, data_validation, data_test = split_8_1_1(data, fold=i)
        
        # save ages, scans, Bo, sex and etiv and drop it from the data. 
        ages_all_train, ages_all_val, ages_all_test = data_train['Age'].values, data_validation['Age'].values, data_test['Age'].values
        scans_all_train, scans_all_val, scans_all_test = data_train['Escaner'].values, data_validation['Escaner'].values, data_test['Escaner'].values
        Bo_all_train, Bo_all_val, Bo_all_test = data_train['Bo'].values, data_validation['Bo'].values, data_test['Bo'].values
        sex_all_train, sex_all_val, sex_all_test = data_train['M/F'].values, data_validation['M/F'].values, data_test['M/F'].values
        etiv_train, etiv_val, etiv_test = data_train['eTIV'].values, data_validation['eTIV'].values, data_test['eTIV'].values

        all_data_train = data_train.drop(['ID', 'Bo', 'M/F', 'Escaner', 'BD', 'Age', 'Patologia'], axis=1)
        all_data_val = data_validation.drop(['ID', 'Bo', 'M/F', 'Escaner', 'BD', 'Age', 'Patologia'], axis=1)
        all_data_test = data_test.drop(['ID', 'Bo', 'M/F', 'Escaner', 'BD', 'Age', 'Patologia'], axis=1)

        features_names = all_data_train.columns.tolist()

        # 2.- erase outliers
        # Save train data to check any errors
        # X_train.to_csv(os.path.join(save_dir, 'X_train_nfeats_'+str(j)+'_split_'+str(i)+'.csv'))
        # X_val.to_csv(os.path.join(save_dir, 'X_val_nfeats_'+str(j)+'_split_'+str(i)+'.csv'))
        # X_test.to_csv(os.path.join(save_dir, 'X_test_nfeats_'+str(j)+'_split_'+str(i)+'.csv'))

        # apply outliers elimination
        X_train, X_val, X_test = outlier_flattening(all_data_train, all_data_val, all_data_test)

        # Transform to array
        X_train, X_val, X_test = X_train.values, X_val.values, X_test.values

        # 3.- normalize data
        X_train, X_val, X_test = normalize_data_min_max(X_train, X_val, X_test, (-1, 1))

        # 4.- Feature Selection
        X_train, X_val, X_test = pd.DataFrame(X_train, columns=features_names), pd.DataFrame(X_val, columns=features_names), \
            pd.DataFrame(X_test, columns=features_names)
        X_train, X_val, X_test, features_names_selected = feature_selection(X_train, X_val, X_test, ages_all_train, j)

        print('[INFO] Number of features: '+str(j))
        print('[INFO] Features selected: \n')
        print(features_names_selected)
        print('[INFO] Shape of training data: '+str(X_train.shape)+'\n')
        features.append(features_names_selected)
        features_tag.append('features_nfeats_'+str(j)+'_split_'+str(i))

        # define lists where to save the results
        lists_SVR = define_lists_svr()
        lists_perceptron = define_lists_cnn()
        lists_RF = define_lists_RF()

        # Execute validation
        # Support vector regressor
        SVRreg = SVR(kernel='linear')
        prediction_SVR_test, prediction_SVR_val, MAEs_and_rs_svr_test, MAEs_and_rs_svr_val =\
            execute_in_val_and_test_SVR(X_train, ages_all_train, X_val, ages_all_val, X_test, ages_all_test, lists_SVR, SVRreg, j, i, save_dir)
        prediction_SVR_val['Bo'], prediction_SVR_test['Bo'] = Bo_all_val, Bo_all_test
        prediction_SVR_val['Escaner'], prediction_SVR_test['Escaner'] = scans_all_val, scans_all_test
        results_dataframe_SVR_test = pd.concat([MAEs_and_rs_svr_test, results_dataframe_SVR_test], axis=0)
        results_dataframe_SVR_val = pd.concat([MAEs_and_rs_svr_val, results_dataframe_SVR_val], axis=0)

        # Tab CNN
        model = Perceptron()
        prediction_perceptron_test, prediction_perceptron_val, MAEs_and_rs_perceptron_test, MAEs_and_rs_perceptron_val =\
            execute_in_val_and_test_NN(X_train, ages_all_train, X_val, ages_all_val, X_test, ages_all_test, lists_perceptron, model, j, i, save_dir)
        prediction_perceptron_val['Bo'], prediction_perceptron_test['Bo'] = Bo_all_val, Bo_all_test
        prediction_perceptron_val['Escaner'], prediction_perceptron_test['Escaner'] = scans_all_val, scans_all_test
        results_dataframe_SVR_test = pd.concat([MAEs_and_rs_perceptron_test, results_dataframe_SVR_test], axis=0)
        results_dataframe_perceptron_val = pd.concat([MAEs_and_rs_perceptron_val, results_dataframe_perceptron_val], axis=0)

        # Random Forest
        RFreg = RandomForestRegressor(random_state=42)
        prediction_RF_test, prediction_RF_val, MAEs_and_rs_RF_test, MAEs_and_rs_RF_val =\
            execute_in_val_and_test_RF(X_train, ages_all_train, X_val, ages_all_val, X_test, ages_all_test, lists_RF, RFreg, j, i, save_dir)
        prediction_RF_val['Bo'], prediction_RF_test['Bo'] = Bo_all_val, Bo_all_test
        prediction_RF_val['Escaner'], prediction_RF_test['Escaner'] =scans_all_val, scans_all_test
        results_dataframe_RF_test = pd.concat([MAEs_and_rs_RF_test, results_dataframe_RF_test], axis=0)
        results_dataframe_RF_val = pd.concat([MAEs_and_rs_RF_val, results_dataframe_RF_val], axis=0)
        
        # save predictions into a list
        prediction_SVR_saved_val.append(prediction_SVR_val)
        prediction_perceptron_saved_val.append(prediction_perceptron_val)
        prediction_RandomForest_saved_val.append(prediction_RF_val)

        prediction_SVR_saved_test.append(prediction_SVR_test)
        prediction_perceptron_saved_test.append(prediction_perceptron_test)
        prediction_RandomForest_saved_test.append(prediction_RF_test)

results_dataframe_SVR_val.to_csv(os.path.join(save_dir, 'results_FastSurfer_svr_val.csv'))
results_dataframe_SVR_test.to_csv(os.path.join(save_dir, 'results_FastSurfer_svr_test.csv'))

results_dataframe_perceptron_val.to_csv(os.path.join(save_dir, 'results_FastSurfer_perceptron_val.csv'))
results_dataframe_SVR_test.to_csv(os.path.join(save_dir, 'results_FastSurfer_perceptron_test.csv'))

results_dataframe_RF_val.to_csv(os.path.join(save_dir, 'results_FastSurfer_RF_val.csv'))
results_dataframe_RF_test.to_csv(os.path.join(save_dir, 'results_FastSurfer_RF_test.csv'))

with open(os.path.join(save_dir, 'list_SVR_val.pkl'), 'wb') as f:
    pickle.dump(prediction_SVR_saved_val, f)
with open(os.path.join(save_dir, 'list_perceptron_val.pkl'), 'wb') as f:
    pickle.dump(prediction_perceptron_saved_val, f)
with open(os.path.join(save_dir, 'list_RandomForest_val.pkl'), 'wb') as f:
    pickle.dump(prediction_RandomForest_saved_val, f)

with open(os.path.join(save_dir, 'list_SVR_test.pkl'), 'wb') as f:
    pickle.dump(prediction_SVR_saved_test, f)
with open(os.path.join(save_dir, 'list_perceptron_test.pkl'), 'wb') as f:
    pickle.dump(prediction_perceptron_saved_test, f)
with open(os.path.join(save_dir, 'list_RandomForest_test.pkl'), 'wb') as f:
    pickle.dump(prediction_RandomForest_saved_test, f)

df_Features = pd.DataFrame(list(zip(features_tag, features)), columns =['features_tag', 'features'])
df_Features.to_csv(os.path.join(save_dir, 'df_features.csv'))



