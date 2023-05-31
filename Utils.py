import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectPercentile, mutual_info_regression
import infoselect as inf

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.linear_model import LinearRegression

from scipy import stats

import os
import configparser

import pickle
from pickle import dump


def brain_age_bias_correction(age_val_folds, pred_val_folds, pred_controles, pred_migr_cr, pred_migr_ep):
    slopes, intercepts = [], []
    for i in range(10):
        x = np.array(age_val_folds[i]).reshape((-1, 1))
        y = np.array(pred_val_folds[i])
        model = LinearRegression().fit(x, y)
        intercepts.append(model.intercept_)
        slopes.append(model.coef_[0])

    pred_controles_corrected = [(pred_controles[i] - intercepts[i]) / slopes[i] for i in range(10)]
    pred_migr_cr_corrected = [(pred_migr_cr[i] - intercepts[i]) / slopes[i] for i in range(10)]
    pred_migr_ep_corrected = [(pred_migr_ep[i] - intercepts[i]) / slopes[i] for i in range(10)]

    return pred_controles_corrected, pred_migr_cr_corrected, pred_migr_ep_corrected


def check_split(datos_list, datos_validation, datos_test):

    va_test_check = not(datos_test.equals(datos_validation))
    check_val_train, check_test_train = [], []
    for dataframe in datos_list[2:10]:
        check_val_train.append(not(dataframe.equals(datos_validation)))
        check_test_train.append(not(dataframe.equals(datos_test)))

    return va_test_check and all(check_val_train) and all(check_test_train)


def split_8_1_1(datos, fold):

    # divido los datos en bloques de 10; estńa randomizados por filas antes de entrar aqui
    datos_list = np.array_split(datos, 10)

    datos_test = datos_list[fold]
    if fold == 0:
        datos_validation = datos_list[fold+1]
        datos_list_check = datos_list[2:10]
        datos_train = pd.concat(datos_list_check, axis=0)
        if check_split(datos_list_check, datos_validation, datos_test):
             print('[INFO] datos correctamente dividos')
        else:
            print('[INFO] Comprobar división de los datos')
    elif fold == 1:
        datos_validation = datos_list[fold+1]
        datos_list_check = datos_list[fold+2:10]+[datos_list[fold-1]]
        datos_train = pd.concat(datos_list_check, axis=0)
        if check_split(datos_list_check, datos_validation, datos_test):
             print('[INFO] datos correctamente dividos')
        else:
            print('[INFO] Comprobar división de los datos')
    elif fold == 8:
        datos_validation = datos_list[fold+1]
        datos_list_check = datos_list[0:8]
        datos_train = pd.concat(datos_list_check, axis=0)
        if check_split(datos_list_check, datos_validation, datos_test):
             print('[INFO] datos correctamente dividos')
        else:
            print('[INFO] Comprobar división de los datos')
    elif fold == 9:
        datos_validation = datos_list[0]
        datos_list_check = datos_list[1:9]
        datos_train = pd.concat(datos_list_check, axis=0)
        if check_split(datos_list_check, datos_validation, datos_test):
             print('[INFO] datos correctamente dividos')
        else:
            print('[INFO] Comprobar división de los datos')
    else:
        datos_validation = datos_list[fold+1]
        datos_list_check = datos_list[fold+2:10]+datos_list[0:fold]
        datos_train = pd.concat(datos_list_check, axis=0)
        if check_split(datos_list_check, datos_validation, datos_test):
             print('[INFO] datos correctamente dividos')
        else:
            print('[INFO] Comprobar división de los datos')

    print('[INFO] Shape de los datos de entrenamineto ' + str(datos_train.values.shape))

    return datos_train, datos_validation, datos_test


def cargo_datos_todos():
    # config parser llamo al archivo de configuración
    config_parser = configparser.ConfigParser(allow_no_value=True)
    bindir = os.path.abspath(os.path.dirname(__file__))
    config_parser.read(bindir + "/cfg.cnf")

    # leo los datos
    datos_todos = config_parser.get("DATOS", "datos_todos_full")
    datos_todos = pd.read_csv(datos_todos)
    print("Cargo Datos")
    print(datos_todos.shape)
    datos_todos.sort_values(by=['ID'], inplace=True)

    # elimino dos casos porque en la info demgrafica no esta ni sexo ni edad
    edades_todos = datos_todos['Age']

    datos_todos.drop(['Age'], axis=1, inplace=True)

    return datos_todos, edades_todos

def normalize_data_min_max_2_entries(datos_train, datos_test, range):

    scaler = MinMaxScaler(feature_range=range)
    datos_train = scaler.fit_transform(datos_train)
    datos_test = scaler.transform(datos_test)

    return datos_train, datos_test

def outlier_flattening(datos_train, datos_val, datos_test):
    datos_train_flat = datos_train.copy()
    datos_test_flat = datos_test.copy()
    datos_val_flat = datos_val.copy()

    for col in datos_train.columns:
        if col == 'sexo':
            continue
        else:
            percentiles = datos_train[col].quantile([0.025, 0.975]).values
            datos_train_flat[col] = np.clip(datos_train[col], percentiles[0], percentiles[1])
            datos_val_flat[col] = np.clip(datos_val[col], percentiles[0], percentiles[1])
            datos_test_flat[col] = np.clip(datos_test[col], percentiles[0], percentiles[1])

    return datos_train_flat, datos_val_flat, datos_test_flat

def normalize_data_min_max(datos_train, datos_val, datos_test, range):

    scaler = MinMaxScaler(feature_range=range)
    datos_train = scaler.fit_transform(datos_train)
    datos_val = scaler.transform(datos_val)
    datos_test = scaler.transform(datos_test)

    return datos_train, datos_val, datos_test

def outlier_flattening_2_entries(datos_train, datos_test):
    datos_train_flat = datos_train.copy()
    datos_test_flat = datos_test.copy()

    for col in datos_train.columns:
        if col == 'sexo':
            continue
        else:
            percentiles = datos_train[col].quantile([0.025, 0.975]).values
            datos_train_flat[col] = np.clip(datos_train[col], percentiles[0], percentiles[1])
            datos_test_flat[col] = np.clip(datos_test[col], percentiles[0], percentiles[1])

    return datos_train_flat, datos_test_flat


def save_list(list_to_save, filename):
    with open(filename, 'wb') as file:
        pickle.dump(list_to_save, file)


def outlier_flattening_3_entries(datos_train, datos_val, datos_test):
    datos_train_flat = datos_train.copy()
    datos_test_flat = datos_test.copy()
    datos_val_flat = datos_val.copy()

    for col in datos_train.columns:
        if col == 'sexo':
            continue
        else:
            percentiles = datos_train[col].quantile([0.025, 0.975]).values
            datos_train_flat[col] = np.clip(datos_train[col], percentiles[0], percentiles[1])
            datos_val_flat[col] = np.clip(datos_val_flat[col], percentiles[0], percentiles[1])
            datos_test_flat[col] = np.clip(datos_test[col], percentiles[0], percentiles[1])

    return datos_train_flat, datos_val_flat, datos_test_flat


def normlize_loading_scaler(data, scaler):

    data_norm = scaler.transform(data)

    return data_norm

def outlier_flattening_limits(data, limits):
    datos_flat = data.copy()

    for col in limits['features'].values.tolist():
        datos_flat[col] = np.clip(data[col], limits[limits['features'] == col]['limits'].values[0][0], limits[limits['features'] == col]['limits'].values[0][1])


    return datos_flat


def normalize_data_min_max_2_entries(datos_train, datos_test, range):

    scaler = MinMaxScaler(feature_range=range)
    datos_train = scaler.fit_transform(datos_train)
    datos_test = scaler.transform(datos_test)

    return datos_train, datos_test


def normalize_data_min_max_3_entries(datos_train, datos_val, datos_test, range):

    scaler = MinMaxScaler(feature_range=range)
    datos_train = scaler.fit_transform(datos_train)
    datos_val = scaler.transform(datos_val)
    datos_test = scaler.transform(datos_test)

    return datos_train, datos_val, datos_test

def outliers_y_normalizacion_2_entries(datos_train, datos_test):

    features = datos_train.columns.tolist()
    datos_test = datos_test[features]

    datos_train_flat, datos_test_flat = outlier_flattening_2_entries(datos_train, datos_test)
    datos_train_norm, datos_test_norm = normalize_data_min_max_2_entries(datos_train_flat, datos_test_flat, (-1, 1))

    datos_train_norm = pd.DataFrame(datos_train_norm, columns=features)
    datos_test_norm = pd.DataFrame(datos_test_norm, columns=features)

    return datos_train_norm, datos_test_norm

def outliers_y_normalizacion_3_entries(datos_train, datos_val, datos_test):

    features = datos_train.columns.tolist()
    datos_test = datos_test[features]

    datos_train_flat, datos_val_flat, datos_test_flat = outlier_flattening_3_entries(datos_train, datos_val, datos_test)
    datos_train_norm, datos_val_norm, datos_test_norm = normalize_data_min_max_3_entries(datos_train_flat, datos_val_flat, datos_test_flat, (-1, 1))

    datos_train_norm = pd.DataFrame(datos_train_norm, columns=features)
    datos_val_norm = pd.DataFrame(datos_val_norm, columns=features)
    datos_test_norm = pd.DataFrame(datos_test_norm, columns=features)

    return datos_train_norm, datos_val_norm, datos_test_norm


def feature_selection(data_train, data_val, data_test, ages_train, n_features):

    # select 10 percent best
    sel_2 = SelectPercentile(mutual_info_regression, percentile=10)
    data_train = sel_2.fit_transform(data_train, ages_train)
    data_val = sel_2.transform(data_val)
    data_test = sel_2.transform(data_test)

    data_train = pd.DataFrame(data_train)
    data_train.columns = sel_2.get_feature_names_out()
    data_val = pd.DataFrame(data_val)
    data_val.columns = sel_2.get_feature_names_out()
    data_test = pd.DataFrame(data_test)
    data_test.columns = sel_2.get_feature_names_out()

    # more MI selection
    gmm = inf.get_gmm(data_train.values, ages_train)
    select = inf.SelectVars(gmm, selection_mode='forward')
    select.fit(data_train.values, ages_train, verbose=False)

    # print(select.get_info())
    # select.plot_mi()
    # select.plot_delta()

    data_train_filtered = select.transform(data_train.values, rd=n_features)
    data_val_filtered = select.transform(data_val.values, rd=n_features)
    data_test_filtered = select.transform(data_test.values, rd=n_features)

    indices = select.feat_hist[n_features]
    names_list = data_test.columns.tolist()
    features_names = [names_list[i] for i in indices]

    return data_train_filtered, data_val_filtered, data_test_filtered, features_names

def define_lists_svr():

    # defino listas para guardar los resultados y un dataframe # SVR
    MAE_list_train_SVR, MAE_list_train_unbiased_SVR, r_list_train_SVR, r_list_train_unbiased_SVR, rs_BAG_train_SVR, \
    rs_BAG_train_unbiased_SVR, alfas_SVR, betas_SVR = [], [], [], [], [], [], [], []
    BAG_ChronoAge_df_SVR = pd.DataFrame()

    listas_SVR = [MAE_list_train_SVR, MAE_list_train_unbiased_SVR, r_list_train_SVR,
                  r_list_train_unbiased_SVR, rs_BAG_train_SVR, rs_BAG_train_unbiased_SVR, alfas_SVR,
                  betas_SVR, BAG_ChronoAge_df_SVR, 'SVR']

    return listas_SVR


def define_lists_RF():

    # defino listas para guardar los resultados y un dataframe # RF
    MAE_list_train_RF, MAE_list_train_unbiased_RF, r_list_train_RF, r_list_train_unbiased_RF, rs_BAG_train_RF, \
    rs_BAG_train_unbiased_RF, alfas_RF, betas_RF = [], [], [], [], [], [], [], []
    BAG_ChronoAge_df_RF = pd.DataFrame()

    listas_RF = [MAE_list_train_RF, MAE_list_train_unbiased_RF, r_list_train_RF,
                  r_list_train_unbiased_RF, rs_BAG_train_RF, rs_BAG_train_unbiased_RF, alfas_RF,
                  betas_RF, BAG_ChronoAge_df_RF, 'RF']

    return listas_RF


def define_lists_cnn():

    # defino listas para guardar los resultados y un dataframe # tab_CNN
    MAE_list_train_tab_CNN, MAE_list_train_unbiased_tab_CNN, r_list_train_tab_CNN, r_list_train_unbiased_tab_CNN, rs_BAG_train_tab_CNN, \
    rs_BAG_train_unbiased_tab_CNN, alfas_tab_CNN, betas_tab_CNN = [], [], [], [], [], [], [], []
    BAG_ChronoAge_df_tab_CNN = pd.DataFrame()

    listas_tab_CNN = [MAE_list_train_tab_CNN, MAE_list_train_unbiased_tab_CNN, r_list_train_tab_CNN,
                  r_list_train_unbiased_tab_CNN, rs_BAG_train_tab_CNN, rs_BAG_train_unbiased_tab_CNN, alfas_tab_CNN,
                  betas_tab_CNN, BAG_ChronoAge_df_tab_CNN, 'tab_CNN']

    return listas_tab_CNN


def execute_in_val_and_test_SVR(data_train_filtered, edades_train, data_val_filtered, edades_val, data_test_filtered, edades_test, lista, regresor, n_feats, split, save_dir):

    # identifico en método de regresión
    regresor_used = lista[9]

    # hago el entrenamiento sobre todos los datos de entrenamiento
    regresor.fit(data_train_filtered, edades_train)

    # save the model to disk
    filename = os.path.join(save_dir, 'SVR_nfeats_'+str(n_feats)+'_split_'+str(split)+'.pkl')
    # pickle.dump(regresor, open(filename, 'wb'))

    # Hago la predicción de los casos de test sanos
    pred_val = regresor.predict(data_val_filtered)
    pred_test = regresor.predict(data_test_filtered)

    # Calculo BAG sanos val & test
    BAG_val_sanos = pred_val - edades_val
    BAG_test_sanos = pred_test - edades_test

    # calculo MAE, MAPE y r validation
    MAE_biased_val = mean_absolute_error(edades_val, pred_val)
    MAPE_biased_val = mean_absolute_percentage_error(edades_val, pred_val)
    r_biased_val = stats.pearsonr(edades_val, pred_val)[0]
    r_bag_real_biased_val = stats.pearsonr(BAG_val_sanos, edades_val)[0]

    # calculo MAE, MAPE y r test
    MAE_biased_test = mean_absolute_error(edades_test, pred_test)
    MAPE_biased_test = mean_absolute_percentage_error(edades_test, pred_test)
    r_biased_test = stats.pearsonr(edades_test, pred_test)[0]
    r_bag_real_biased_test = stats.pearsonr(BAG_test_sanos, edades_test)[0]

    # Calculo r MAE para validation
    print('----------- '+regresor_used+' r & MAE val biased -------------')
    print('MAE val: ' + str(MAE_biased_val))
    print('MAPE val: ' + str(MAPE_biased_val))
    print('r val: ' + str(r_biased_val))

    # calculo r biased
    print('--------- '+regresor_used+' Correlación BAG edad real val -------------')
    print('r BAG-edad real val biased: ' + str(r_bag_real_biased_val))
    print('')

    # Calculo r MAE para test
    print('----------- '+regresor_used+' r & MAE test biased -------------')
    print('MAE test: ' + str(MAE_biased_test))
    print('MAPE test: ' + str(MAPE_biased_test))
    print('r test: ' + str(r_biased_test))

    # calculo r biased test
    print('--------- '+regresor_used+' Correlación BAG edad real test -------------')
    print('r BAG-edad real test biased: ' + str(r_bag_real_biased_test))
    print('')

    # Figura concordancia entre predichas y reales con reg lineal
    prediction_and_real_data_test = pd.DataFrame(list(zip(edades_test, pred_test)), columns=['edades_test', 'pred_test'])
    prediction_and_real_data_val = pd.DataFrame(list(zip(edades_val, pred_val)), columns=['edades_val', 'pred_val'])
    # sns.regplot(data=prediction_and_real_data, x="edades_test", y="pred_test", line_kws={"color": "red"})
    # plt.title('Edad real vs edad predicha')
    # plt.ylabel("Edad predicha")
    # plt.xlabel("Edad real")
    # plt.show()

    MAEs_and_rs_test = pd.DataFrame(list(zip([MAE_biased_test], [r_biased_test], [r_bag_real_biased_test])),
                                            columns=['MAE_biased_test', 'r_biased_test', 'r_bag_real_biased_test'])

    MAEs_and_rs_val = pd.DataFrame(list(zip([MAE_biased_val], [r_biased_val], [r_bag_real_biased_val])),
                                            columns=['MAE_biased_val', 'r_biased_val', 'r_bag_real_biased_val'])

    # results = permutation_importance(regresor, data_train_filtered, edades_train, scoring='neg_mean_absolute_error', n_jobs=-1)

    return prediction_and_real_data_test, prediction_and_real_data_val, MAEs_and_rs_test, MAEs_and_rs_val




def execute_in_val_and_test_NN(data_train_filtered, edades_train, data_val_filtered, edades_val, data_test_filtered, edades_test, lista, regresor, n_features, split, save_dir):

    # identifico en método de regresión
    regresor_used = lista[9]

    # hago el entrenamiento sobre todos los datos de entrenamiento
    regresor.fit(data_train_filtered, edades_train, n_features, 16,  lr=1e-3, weight_decay=1e-6, validation_size=0.2)

    # save the model to disk
    filename = os.path.join(save_dir, 'MLP_nfeats_' + str(n_features) + '_split_' + str(split) + '.pkl')
    # pickle.dump(regresor, open(filename, 'wb'))

    # Hago la predicción de los casos de test sanos
    pred_val = regresor.predict(data_val_filtered)
    pred_test = regresor.predict(data_test_filtered)

    # Calculo BAG sanos val & test
    BAG_val_sanos = pred_val - edades_val
    BAG_test_sanos = pred_test - edades_test

    # calculo MAE, MAPE y r validation
    MAE_biased_val = mean_absolute_error(edades_val, pred_val)
    MAPE_biased_val = mean_absolute_percentage_error(edades_val, pred_val)
    r_biased_val = stats.pearsonr(edades_val, pred_val)[0]
    r_bag_real_biased_val = stats.pearsonr(BAG_val_sanos, edades_val)[0]

    # calculo MAE, MAPE y r test
    MAE_biased_test = mean_absolute_error(edades_test, pred_test)
    MAPE_biased_test = mean_absolute_percentage_error(edades_test, pred_test)
    r_biased_test = stats.pearsonr(edades_test, pred_test)[0]
    r_bag_real_biased_test = stats.pearsonr(BAG_test_sanos, edades_test)[0]

    # Calculo r MAE para validation
    print('----------- ' + regresor_used + ' r & MAE Val biased -------------')
    print('MAE val: ' + str(MAE_biased_val))
    print('MAPE val: ' + str(MAPE_biased_val))
    print('r val: ' + str(r_biased_val))

    # calculo r biased
    print('--------- ' + regresor_used + ' Correlación BAG edad real train -------------')
    print('r BAG-edad real val biased: ' + str(r_bag_real_biased_val))
    print('')

    # Calculo r MAE para test
    print('----------- ' + regresor_used + ' r & MAE test biased -------------')
    print('MAE test: ' + str(MAE_biased_test))
    print('MAPE test: ' + str(MAPE_biased_test))
    print('r test: ' + str(r_biased_test))

    # calculo r biased test
    print('--------- ' + regresor_used + ' Correlación BAG edad real test -------------')
    print('r BAG-edad real test biased: ' + str(r_bag_real_biased_test))
    print('')

    # Figura concordancia entre predichas y reales con reg lineal
    prediction_and_real_data_test = pd.DataFrame(list(zip(edades_test, pred_test)), columns=['edades_test', 'pred_test'])
    prediction_and_real_data_val = pd.DataFrame(list(zip(edades_val, pred_val)), columns=['edades_val', 'pred_val'])
    # sns.regplot(data=prediction_and_real_data, x="edades_test", y="pred_test", line_kws={"color": "red"})
    # plt.title('Edad real vs edad predicha')
    # plt.ylabel("Edad predicha")
    # plt.xlabel("Edad real")
    # plt.show()

    MAEs_and_rs_test = pd.DataFrame(list(zip([MAE_biased_test], [r_biased_test], [r_bag_real_biased_test])),
                                    columns=['MAE_biased_test', 'r_biased_test', 'r_bag_real_biased_test'])

    MAEs_and_rs_val = pd.DataFrame(list(zip([MAE_biased_val], [r_biased_val], [r_bag_real_biased_val])),
                                   columns=['MAE_biased_val', 'r_biased_val', 'r_bag_real_biased_val'])

    # results = permutation_importance(regresor, data_train_filtered, edades_train, scoring='neg_mean_absolute_error', n_jobs=-1)

    return prediction_and_real_data_test, prediction_and_real_data_val, MAEs_and_rs_test, MAEs_and_rs_val


def execute_in_val_and_test_RF(data_train_filtered, edades_train, data_val_filtered, edades_val,
                                data_test_filtered, edades_test, lista, regresor, n_feats, split, save_dir):
    # identifico en método de regresión
    regresor_used = lista[9]

    # hago el entrenamiento sobre todos los datos de entrenamiento
    regresor.fit(data_train_filtered, edades_train)

    # save the model to disk
    filename = os.path.join(save_dir, 'RF_nfeats_' + str(n_feats) + '_split_' + str(split) + '.pkl')
    # pickle.dump(regresor, open(filename, 'wb'))

    # Hago la predicción de los casos de test sanos
    pred_val = regresor.predict(data_val_filtered)
    pred_test = regresor.predict(data_test_filtered)

    # Calculo BAG sanos val & test
    BAG_val_sanos = pred_val - edades_val
    BAG_test_sanos = pred_test - edades_test

    # calculo MAE, MAPE y r validation
    MAE_biased_val = mean_absolute_error(edades_val, pred_val)
    MAPE_biased_val = mean_absolute_percentage_error(edades_val, pred_val)
    r_biased_val = stats.pearsonr(edades_val, pred_val)[0]
    r_bag_real_biased_val = stats.pearsonr(BAG_val_sanos, edades_val)[0]

    # calculo MAE, MAPE y r test
    MAE_biased_test = mean_absolute_error(edades_test, pred_test)
    MAPE_biased_test = mean_absolute_percentage_error(edades_test, pred_test)
    r_biased_test = stats.pearsonr(edades_test, pred_test)[0]
    r_bag_real_biased_test = stats.pearsonr(BAG_test_sanos, edades_test)[0]

    # Calculo r MAE para validation
    print('----------- ' + regresor_used + ' r & MAE val biased -------------')
    print('MAE val: ' + str(MAE_biased_val))
    print('MAPE val: ' + str(MAPE_biased_val))
    print('r val: ' + str(r_biased_val))

    # calculo r biased
    print('--------- ' + regresor_used + ' Correlación BAG edad real train -------------')
    print('r BAG-edad real val biased: ' + str(r_bag_real_biased_val))
    print('')

    # Calculo r MAE para test
    print('----------- ' + regresor_used + ' r & MAE test biased -------------')
    print('MAE test: ' + str(MAE_biased_test))
    print('MAPE test: ' + str(MAPE_biased_test))
    print('r test: ' + str(r_biased_test))

    # calculo r biased test
    print('--------- ' + regresor_used + ' Correlación BAG edad real test -------------')
    print('r BAG-edad real test biased: ' + str(r_bag_real_biased_test))
    print('')

    # Figura concordancia entre predichas y reales con reg lineal
    prediction_and_real_data_test = pd.DataFrame(list(zip(edades_test, pred_test)), columns=['edades_test', 'pred_test'])
    prediction_and_real_data_val = pd.DataFrame(list(zip(edades_val, pred_val)), columns=['edades_val', 'pred_val'])
    # sns.regplot(data=prediction_and_real_data, x="edades_test", y="pred_test", line_kws={"color": "red"})
    # plt.title('Edad real vs edad predicha')
    # plt.ylabel("Edad predicha")
    # plt.xlabel("Edad real")
    # plt.show()

    MAEs_and_rs_test = pd.DataFrame(list(zip([MAE_biased_test], [r_biased_test], [r_bag_real_biased_test])),
                                    columns=['MAE_biased_test', 'r_biased_test', 'r_bag_real_biased_test'])

    MAEs_and_rs_val = pd.DataFrame(list(zip([MAE_biased_val], [r_biased_val], [r_bag_real_biased_val])),
                                   columns=['MAE_biased_val', 'r_biased_val', 'r_bag_real_biased_val'])

    # results = permutation_importance(regresor, data_train_filtered, edades_train, scoring='neg_mean_absolute_error', n_jobs=-1)

    return prediction_and_real_data_test, prediction_and_real_data_val, MAEs_and_rs_test, MAEs_and_rs_val


