import os
import configparser
from utils import *

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from MLP_1_layer import *
from torch import nn
from torch import optim
import torch

datos_sanos_oasis, edades_OASIS = cargo_datos_OASIS()
edades_OASIS = edades_OASIS.reset_index(drop=True)
datos_sanos_oasis.reset_index(inplace=True, drop=True)
datos_sanos_oasis = datos_sanos_oasis.drop(['ID', 'Bo'], axis=1)

# config parser llamo al archivo de configuración
config_parser = configparser.ConfigParser(allow_no_value=True)
bindir = os.path.abspath(os.path.dirname(__file__))
config_parser.read(bindir + "/cfg.cnf")

datos_train, datos_test, edades_train, edades_test = train_test_split(datos_sanos_oasis, edades_OASIS, test_size=0.2, random_state=0)
features_names_RFE = ['Volume_mm3_Left-Inf-Lat-Vent', 'normMean_Left-Pallidum', 'normMean_3rd-Ventricle',
                      'Volume_mm3_Left-Accumbens-area', 'normMean_Left-choroid-plexus',
                      'Volume_mm3_Right-Putamen', 'normMean_Right-Accumbens-area', 'Volume_mm3_CC_Mid_Anterior',
                      'ThickAvg_lh_parsopercularis', 'GrayVol_lh_postcentral',
                      'ThickStd_lh_rostralmiddlefrontal', 'ThickStd_lh_superiorfrontal', 'ThickAvg_rh_parsopercularis',
                      'Mean_lh_wg_pct_paracentral',
                      'Mean_lh_wg_pct_parstriangularis', 'Mean_lh_wg_pct_superiorfrontal',
                      'Mean_rh_wg_pct_inferiorparietal', 'Mean_rh_wg_pct_precentral',
                      'Mean_rh_wg_pct_rostralmiddlefrontal', 'normMinwm-lh-parstriangularis',
                      'normStdDevwm-lh-superiorfrontal', 'normMeanwm-lh-transversetemporal',
                      'normStdDevwm-lh-insula', 'normMeanwm-rh-cuneus', 'normMeanLeft-UnsegmentedWhiteMatter']
datos_train = datos_train[features_names_RFE]
datos_test = datos_test[features_names_RFE]
datos_train_flat, datos_test_flat = outlier_flattening(datos_train, datos_test)
datos_train_norm, datos_test_norm = normalize_data_min_max(datos_train_flat, datos_test_flat, (-1, 1))


model = Perceptron()
model.fit(datos_train_norm, edades_train.values)
pred = model.predict(datos_test_norm)

print('MAE '+str(mean_absolute_error(edades_test.values, pred)))

print('fin')



