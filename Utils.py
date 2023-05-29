import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


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