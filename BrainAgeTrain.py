from scipy.stats import ks_2samp
from sklearn.model_selection import KFold

from sklearn.preprocessing import LabelEncoder
from MultilayerPerceptron.MLP_1_layer import *

from neuroHarmonize import harmonizationLearn
import configparser

from Utils import *
import pickle

datos_todos, edades_todos = cargo_datos_todos()
edades_todos = edades_todos.reset_index(drop=True)
datos_todos.reset_index(inplace=True, drop=True)

# save dir
config_parser = configparser.ConfigParser(allow_no_value=True)
bindir = os.path.abspath(os.path.dirname(__file__))
config_parser.read(bindir + "/cfg.cnf")

save_dir = config_parser.get("RESULTADOS", "Resultados_Brain_Age_ComBatGAM_todo_miRNA")

# Añado la edad al dataframe para facilitarme el proceso
datos_todos['Age'] = edades_todos

# filtro los datos que me interesan; No NKI-Rockland; No CoRR; Sólo morfo; solo edades entre 18 y 65
# datos_todos = datos_todos[datos_todos['BD'] != 'NKI']
# datos_todos = datos_todos[datos_todos['BD'] != 'CoRR']

datos_todos = datos_todos[datos_todos['Patologia'] != 'COVID']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Migraña_cefalea_Resto']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Migraña_cefalea_2as']
# datos_todos = datos_todos[datos_todos['Patologia'] != 'Control_seleccionado']
# datos_todos = datos_todos[datos_todos['Patologia'] != 'Migraña_EP_elegidos']
# datos_todos = datos_todos[datos_todos['Patologia'] != 'Migraña_CR_elegidos']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Depresion']
datos_todos = datos_todos[datos_todos['Patologia'] != 'dolor_de_cabeza_repetido_o_severo']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Migraña_NKI']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Convulsions_seizures']
datos_todos = datos_todos[datos_todos['Patologia'] != 'epilepsia']
datos_todos = datos_todos[datos_todos['Patologia'] != 'TBI']
datos_todos = datos_todos[datos_todos['Patologia'] != 'defectos_nacimiento']
datos_todos = datos_todos[datos_todos['Patologia'] != 'intox_plomo']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Problema_lenguaje']
datos_todos = datos_todos[datos_todos['Patologia'] != 'tics_vocales']
datos_todos = datos_todos[datos_todos['Patologia'] != 'tics_motores']
datos_todos = datos_todos[datos_todos['Patologia'] != 'dislexia']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Prob_aprendizaje']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Hiperactividad']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Prob_atención']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Autismo']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Sonambulismo']
datos_todos = datos_todos[datos_todos['Patologia'] != 'moja_cama']
datos_todos = datos_todos[datos_todos['Patologia'] != 'prob_intensitnal']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Cancer']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Ataque_corazon']
datos_todos = datos_todos[datos_todos['Patologia'] != 'prob_coronario']
datos_todos = datos_todos[datos_todos['Patologia'] != 'prob_valvs']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Hipercolesterolemia']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Hipertension']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Hipotension']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Arritmia']
datos_todos = datos_todos[datos_todos['Patologia'] != 'ACV']
datos_todos = datos_todos[datos_todos['Patologia'] != 'IBS']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Crohn']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Colitis']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Reflujo']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Hepatitis']
datos_todos = datos_todos[datos_todos['Patologia'] != 'DB-1']
datos_todos = datos_todos[datos_todos['Patologia'] != 'DB-2']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Hiper_th']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Hipo_th']
datos_todos = datos_todos[datos_todos['Patologia'] != 'HIV']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Artritis']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Osteoporosis']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Enfisema']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Acne_sev']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Psoriasis']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Fatiga_Cronica']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Fibromialgia']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Bipolar']
datos_todos = datos_todos[datos_todos['Patologia'] != 'MCI']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Intento_Suicidio']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Esquizofrenia']
datos_todos = datos_todos[datos_todos['Patologia'] != 'OCD']
datos_todos = datos_todos[datos_todos['Patologia'] != 'ansiedad_social']
datos_todos = datos_todos[datos_todos['Patologia'] != 'PTSD']
datos_todos = datos_todos[datos_todos['Patologia'] != 'ataques_pánico']
datos_todos = datos_todos[datos_todos['Patologia'] != 'otros_ansiedad']
datos_todos = datos_todos[datos_todos['Patologia'] != 'ADHD']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Anorexia']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Bulimia']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Alzheimer']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Huntington']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Meningitis']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Esclerosis_multiple']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Parkinson']
datos_todos = datos_todos[datos_todos['Patologia'] != 'Enfermedad_de_Lyme']
datos_todos = datos_todos[datos_todos['Age'] >= 18]
datos_todos = datos_todos[datos_todos['Age'] <= 60]

# Quito dos escáneres que tienen una n muy baja, y me joden la armonización
datos_todos = datos_todos[datos_todos['Escaner'] != 'CoRR_LMU_3_Siemens_TrioTim']
datos_todos = datos_todos[datos_todos['Escaner'] != 'CoRR_Utah_1_Siemens_TrioTim']
datos_todos = datos_todos[datos_todos['BD'] != 'NKI']

lista_maquinas_todos = datos_todos['Escaner'].values
etiv_todos = datos_todos['eTIV'].values
edades_todos = datos_todos['Age']
bo_todos = datos_todos['Bo'].values
sex_todos = datos_todos['M/F'].values
IDs = datos_todos['ID'].values
BDs = datos_todos['BD'].values
Patologia = datos_todos['Patologia'].values
datos_todos = datos_todos.drop(['ID', 'Bo', 'M/F', 'Escaner', 'BD', 'Patologia', 'Age'], axis=1)

# armonizo las características con la edad como covariable usando ComBat
# Hago la armonización de ComBat antes
columns_to_drop = [96, 97, 98, 99, 100, 101, 180, 181, 182, 183, 184, 185, 192, 193, 194, 195, 196, 197, 204, 205, 206,
                   207, 208, 209, 210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226,
                   227, 228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239]
features_to_drop = datos_todos.columns[columns_to_drop]
datos_todos = datos_todos.drop(features_to_drop, axis=1)
features_names = datos_todos.columns.tolist()

# incluyo los escáneres y el sexo como LabelEncoder
LE = LabelEncoder()
datos_maquinas_num = pd.DataFrame(LE.fit_transform(lista_maquinas_todos))
LE_sex = LabelEncoder()
datos_sex_num = pd.DataFrame(LE_sex.fit_transform(sex_todos))

# monto los datos de entrada y la matriz de covariables
datos_array = datos_todos.values
d = {'SITE': datos_maquinas_num.values.tolist(), 'SEX': np.squeeze(datos_sex_num.values).tolist(), 'ETIV': etiv_todos.tolist(), 'AGE':edades_todos.values.tolist()}
covars = pd.DataFrame(data=d)
my_model, datos = harmonizationLearn(datos_array, covars)

datos = pd.DataFrame(datos, columns = features_names)
datos['Age'] = edades_todos.values
datos['Escaner'] = lista_maquinas_todos
datos['Bo'] = bo_todos
datos['M/F'] = sex_todos
datos['eTIV'] = etiv_todos
datos['ID'] = IDs
datos['BD'] = BDs
datos['Patologia'] = Patologia

# quito los datos de migraña armonizados y los guardo
datos_controles = datos[datos['Patologia'] == 'Control_seleccionado']
datos = datos[datos['Patologia'] != 'Control_seleccionado']
datos_controles.to_csv(os.path.join(save_dir, 'Controles_harmo.csv'))

print('Check media valores edad de controles:')
print(np.mean(datos_controles['Age'].values))

datos_migrana_EP = datos[datos['Patologia'] == 'Migraña_EP_elegidos']
datos = datos[datos['Patologia'] != 'Migraña_EP_elegidos']
datos_migrana_EP.to_csv(os.path.join(save_dir, 'migraña_ep_harmo.csv'))

datos_migrana_CR = datos[datos['Patologia'] == 'Migraña_CR_elegidos']
datos = datos[datos['Patologia'] != 'Migraña_CR_elegidos']
datos_migrana_CR.to_csv(os.path.join(save_dir, 'migraña_cr_harmo.csv'))

# randomizo el orden de las filas y reseteo índices
datos = datos.sample(frac=1, random_state=42).reset_index(drop=True)

# resultado deltest KS p valor
ks_result, features, features_tag = [], [], []
Results_dataframe_SVR_test = pd.DataFrame()
Results_dataframe_perceptron_test = pd.DataFrame()
Results_dataframe_RF_test = pd.DataFrame()
Results_dataframe_SVR_val = pd.DataFrame()
Results_dataframe_perceptron_val = pd.DataFrame()
Results_dataframe_RF_val = pd.DataFrame()

MAE_val, MAE_test, r_test = [], [], []

prediction_SVR_saved_test, prediction_perceptron_saved_test, prediction_RandomForest_saved_test = [], [], []
prediction_SVR_saved_val, prediction_perceptron_saved_val, prediction_RandomForest_saved_val = [], [], []
for j in [20, 30, 40]:
    for i in range(0, 10, 1):

        datos_train, datos_validation, datos_test = split_8_1_1(datos, fold=i)

        edades_todos_train = datos_train['Age'].values
        edades_todos_val = datos_validation['Age'].values
        edades_todos_test = datos_test['Age'].values

        maquinas_todos_train = datos_train['Escaner'].values
        maquinas_todos_val = datos_validation['Escaner'].values
        maquinas_todos_test = datos_test['Escaner'].values

        Bo_todos_train = datos_train['Bo'].values
        Bo_todos_val = datos_validation['Bo'].values
        Bo_todos_test = datos_test['Bo'].values

        sex_todos_train = datos_train['M/F'].values
        sex_todos_val = datos_validation['M/F'].values
        sex_todos_test = datos_test['M/F'].values

        etiv_train = datos_train['eTIV'].values
        etiv_val = datos_validation['eTIV'].values
        etiv_test = datos_test['eTIV'].values

        datos_todos_train = datos_train.drop(['ID', 'Bo', 'M/F', 'Escaner', 'BD', 'Age', 'Patologia'], axis=1)
        datos_todos_val = datos_validation.drop(['ID', 'Bo', 'M/F', 'Escaner', 'BD', 'Age', 'Patologia'], axis=1)
        datos_todos_test = datos_test.drop(['ID', 'Bo', 'M/F', 'Escaner', 'BD', 'Age', 'Patologia'], axis=1)

        features_morphological = datos_todos_train.columns.tolist()

        # test de Kolmogorov-Smirnov para ver si las distribuciones son iguales; tiene que salir estadísticamente no siginificativo
        ks_test = ks_2samp(edades_todos_train, edades_todos_test)
        print('test de Kolmogorov-Smirnov para edades train-test')
        print('si es mayor de 0.05 no puedo descartar que sean iguales: '+str(ks_test[1]))
        ks_result.append(ks_test[1])

        # 2.- elimino outliers
        # transformo a dataframe
        X_train = pd.DataFrame(datos_todos_train, columns=features_morphological)
        X_val = pd.DataFrame(datos_todos_val, columns=features_morphological)
        X_test = pd.DataFrame(datos_todos_test, columns=features_morphological)

        # guardo los datos de entreno para usarlos luego en el 2º test, validación y test
        X_train.to_csv(os.path.join(save_dir, 'X_train_nfeats_'+str(j)+'_split_'+str(i)+'.csv'))
        X_val.to_csv(os.path.join(save_dir, 'X_val_nfeats_'+str(j)+'_split_'+str(i)+'.csv'))
        X_test.to_csv(os.path.join(save_dir, 'X_test_nfeats_'+str(j)+'_split_'+str(i)+'.csv'))

        # aplico la eliminación de outliers
        X_train, X_val, X_test = outlier_flattening(X_train, X_val, X_test)

        # transformo a array
        X_train = X_train.values
        X_test = X_test.values
        X_val = X_val.values

        # 3.- normalizo los datos OJO LA NORMALIZACION QUE CON Z NORM O CON 0-1 PUEDE VARIAR EL RESULTADO BASTANTE!
        X_train, X_val, X_test = normalize_data_min_max(X_train, X_val, X_test, (-1, 1))

        # 4.- Feature Selection
        X_train = pd.DataFrame(X_train, columns=features_morphological)
        X_val = pd.DataFrame(X_val, columns=features_morphological)
        X_test = pd.DataFrame(X_test, columns=features_morphological)
        X_train, X_val, X_test, features_names_SFS = feature_selection_compossite_MI_with_val(
            X_train, X_val, X_test, edades_todos_train, j)

        print('[INFO] Número de características: '+str(j))
        print('[INFO] Features selected: \n')
        print(features_names_SFS)
        print('shape de los datos de entrenamineto: '+str(X_train.shape)+'\n')
        features.append(features_names_SFS)
        features_tag.append('features_nfeats_'+str(j)+'_split_'+str(i))

        # 6.- entreno el regresor 10xCV
        kf = KFold(n_splits=10, random_state=i, shuffle=True)

        # defino listas donde van a ir los resultados
        listas_SVR = define_listas_svr()
        listas_perceptron = define_listas_cnn()
        listas_RF = define_listas_RF()

        # Ejecuto en validación
        # Support vector regressor
        SVRreg = SVR(kernel='linear')
        prediction_SVR_test, prediction_SVR_val, MAEs_and_rs_svr_test, MAEs_and_rs_svr_val = execute_in_val_and_test_SVR_con_CoRR(X_train, edades_todos_train, X_val, edades_todos_val, X_test, edades_todos_test, listas_SVR, SVRreg, j, i, save_dir)
        prediction_SVR_test['Bo'] = Bo_todos_test
        prediction_SVR_test['Escaner'] = maquinas_todos_test
        prediction_SVR_val['Bo'] = Bo_todos_val
        prediction_SVR_val['Escaner'] = maquinas_todos_val
        Results_dataframe_SVR_test = pd.concat([MAEs_and_rs_svr_test, Results_dataframe_SVR_test], axis=0)
        Results_dataframe_SVR_val = pd.concat([MAEs_and_rs_svr_val, Results_dataframe_SVR_val], axis=0)

        # Tab CNN
        model = Perceptron()
        prediction_perceptron_test, prediction_perceptron_val, MAEs_and_rs_perceptron_test, MAEs_and_rs_perceptron_val = execute_in_val_and_test_NN_con_CoRR(X_train, edades_todos_train, X_val, edades_todos_val, X_test, edades_todos_test, listas_perceptron, model, j, i, save_dir)
        prediction_perceptron_test['Bo'] = Bo_todos_test
        prediction_perceptron_test['Escaner'] = maquinas_todos_test
        prediction_perceptron_val['Bo'] = Bo_todos_val
        prediction_perceptron_val['Escaner'] = maquinas_todos_val
        Results_dataframe_perceptron_test = pd.concat([MAEs_and_rs_perceptron_test, Results_dataframe_perceptron_test], axis=0)
        Results_dataframe_perceptron_val = pd.concat([MAEs_and_rs_perceptron_val, Results_dataframe_perceptron_val], axis=0)

        # Random Forest
        RFreg = RandomForestRegressor(random_state=42)
        prediction_RF_test, prediction_RF_val, MAEs_and_rs_RF_test, MAEs_and_rs_RF_val = execute_in_val_and_test_RF_con_CoRR(X_train, edades_todos_train, X_val, edades_todos_val, X_test, edades_todos_test, listas_RF, RFreg, j, i, save_dir)
        prediction_RF_test['Bo'] = Bo_todos_test
        prediction_RF_test['Escaner'] = maquinas_todos_test
        prediction_RF_val['Bo'] = Bo_todos_val
        prediction_RF_val['Escaner'] = maquinas_todos_val
        Results_dataframe_RF_test = pd.concat([MAEs_and_rs_RF_test, Results_dataframe_RF_test], axis=0)
        Results_dataframe_RF_val = pd.concat([MAEs_and_rs_RF_val, Results_dataframe_RF_val], axis=0)

        prediction_SVR_saved_val.append(prediction_SVR_val)
        prediction_perceptron_saved_val.append(prediction_perceptron_val)
        prediction_RandomForest_saved_val.append(prediction_RF_val)

        prediction_SVR_saved_test.append(prediction_SVR_test)
        prediction_perceptron_saved_test.append(prediction_perceptron_test)
        prediction_RandomForest_saved_test.append(prediction_RF_test)

# figura_de_MAE_y_MAPE_por_rango(prediction_SVR_test)
# print(Results_dataframe_SVR)
Results_dataframe_SVR_val.to_csv(os.path.join(save_dir,'results_FastSurfer_svr_val.csv'))
Results_dataframe_SVR_test.to_csv(os.path.join(save_dir,'results_FastSurfer_svr_test.csv'))

Results_dataframe_perceptron_val.to_csv(os.path.join(save_dir,'results_FastSurfer_perceptron_val.csv'))
Results_dataframe_perceptron_test.to_csv(os.path.join(save_dir,'results_FastSurfer_perceptron_test.csv'))

Results_dataframe_RF_val.to_csv(os.path.join(save_dir,'results_FastSurfer_RF_val.csv'))
Results_dataframe_RF_test.to_csv(os.path.join(save_dir,'results_FastSurfer_RF_test.csv'))

with open(os.path.join(save_dir, 'lista_prueba_SVR_test.pkl'), 'wb') as f:
    pickle.dump(prediction_SVR_saved_test, f)
with open(os.path.join(save_dir, 'lista_prueba_perceptron_test.pkl'), 'wb') as f:
    pickle.dump(prediction_perceptron_saved_test, f)
with open(os.path.join(save_dir, 'lista_prueba_RandomForest_test.pkl'), 'wb') as f:
    pickle.dump(prediction_RandomForest_saved_test, f)

with open(os.path.join(save_dir, 'lista_prueba_SVR_val.pkl'), 'wb') as f:
    pickle.dump(prediction_SVR_saved_val, f)
with open(os.path.join(save_dir, 'lista_prueba_perceptron_val.pkl'), 'wb') as f:
    pickle.dump(prediction_perceptron_saved_val, f)
with open(os.path.join(save_dir, 'lista_prueba_RandomForest_val.pkl'), 'wb') as f:
    pickle.dump(prediction_RandomForest_saved_val, f)

df_Features = pd.DataFrame(list(zip(features_tag, features)), columns =['features_tag', 'features'])
df_Features.to_csv(os.path.join(save_dir, 'df_features_con_CoRR.csv'))

print(features)
print(ks_result)


