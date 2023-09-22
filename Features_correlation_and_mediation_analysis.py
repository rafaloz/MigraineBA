import os
import configparser

import pickle
import numpy as np
import pandas as pd
import pingouin as pg

# config parser llamo al archivo de configuración
config_parser = configparser.ConfigParser(allow_no_value=True)
bindir = os.path.abspath(os.path.dirname(__file__))
config_parser.read(bindir + "/cfg.cnf")

folderApplication = config_parser.get("DATA", "DataAplication")

print('#################################################################')
print('[INFO] Starting mediation analysis...')
print('#################################################################', end='\n\n')

with open(os.path.join(folderApplication, 'Demographics_and_Clinical/MediationAnalysis_variables.pkl'), 'rb') as f:
    (freq_cef_EM, freq_cef_CM, BrainVol_EM, BrainVol_CM,
     BAG_EM, BAG_CM, sex_EM, sex_CM, age_EM, age_CM) = pickle.load(f)

# Concatenate and Standardize variables // Choose your variables. BAG calculated with the combined MLP40 model
X = np.concatenate((freq_cef_EM, freq_cef_CM))
M = np.concatenate((BrainVol_EM, BrainVol_CM))
Y = np.concatenate((BAG_EM, BAG_CM))
sex = np.concatenate((sex_EM, sex_CM))
age = np.concatenate((age_EM, age_CM))

variables = [X, M, Y]
standardized_vars = [(var - var.mean()) / var.std() for var in variables]

# Create a DataFrame using list comprehension for sex
sexos_cat = [1 if value == 'F' else 0 for value in sex]

df = pd.DataFrame({
    'X': standardized_vars[0],
    'M': standardized_vars[1],
    'Y': standardized_vars[2],
    'sex': sexos_cat,
    'age': age
})

# Perform mediation analysis using bootstrapping and bias-correction
print('################ Results Pingouin mediation analysis #################')
mediation = pg.mediation_analysis(data=df, x='X', m='M', y='Y', covar=['age', 'sex'], n_boot=10000, alpha=0.05)
print(mediation, end='\n\n')

print('############################### END SCRIPT ###############################')
print('##########################################################################')