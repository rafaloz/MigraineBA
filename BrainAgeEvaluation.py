import matplotlib.pyplot as plt
import pandas as pd
import pickle

from Utils import *

import os
import configparser

from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression

from statsmodels.stats import weightstats as stests

from pingouin import ancova, welch_anova

from scipy.stats import f_oneway
from scipy.stats import kstest, kruskal, levene, chi2_contingency, mannwhitneyu
from scipy import stats

import seaborn as sns


# config parser llamo al archivo de configuración
config_parser = configparser.ConfigParser(allow_no_value=True)
bindir = os.path.abspath(os.path.dirname(__file__))
config_parser.read(bindir + "/cfg.cnf")

folders = config_parser.get("RESULTADOS", "Resultados_Brain_Age")

datos_todos, edades_todos = cargo_datos_todos()
edades_todos = edades_todos.reset_index(drop=True)
datos_todos.reset_index(inplace=True, drop=True)
datos_todos['Age'] = edades_todos

# cargo los datos
datos_migr_ep = os.path.join(folders, "Resultados_paper", "miRNA", "Resultados_ComBatGAM_all_paper", "migraña_ep_harmo.csv")
datos_migr_ep = pd.read_csv(datos_migr_ep).iloc[:, 1:]

edad_migr_ep = datos_migr_ep['Age'].values
sex_migr_ep = datos_migr_ep['M/F'].values

# datos_migr_cr = config_parser.get("DATOS", "datos_migr_cr")
datos_migr_cr = os.path.join(folders, "Resultados_paper", "miRNA", "Resultados_ComBatGAM_all_paper", "migraña_cr_harmo.csv")
datos_migr_cr = pd.read_csv(datos_migr_cr).iloc[:, 1:]
edad_migr_cr = datos_migr_cr['Age'].values
sex_migr_cr = datos_migr_cr['M/F'].values

# datos_control_sel = config_parser.get("DATOS", "datos_control_sel")
datos_control_sel = os.path.join(folders, "Resultados_paper", "miRNA", "Resultados_ComBatGAM_all_paper", "Controles_harmo.csv")
datos_control_sel = pd.read_csv(datos_control_sel).iloc[:, 1:]
edad_controles_sel = datos_control_sel['Age'].values
sex_control_sel = datos_control_sel['M/F'].values

datos_cron_save = datos_migr_cr.iloc[:, -7:]
datos_ep_save = datos_migr_ep.iloc[:, -7:]

print('Mediana y media de edad del conjunto de control')
print(np.median(edad_controles_sel))
print(np.mean(edad_controles_sel))

dataset_2 = pd.concat([datos_migr_ep, datos_migr_cr, datos_control_sel], axis=0)

folder_modelos_con_CoRR = os.path.join(folders, "Resultados_paper", "miRNA", "Resultados_ComBatGAM_all_paper")

# cargo los modelos y las features; Mejor resultado validación MLP 40 features;
df_features = pd.read_csv(os.path.join(folder_modelos_con_CoRR, 'df_features_con_CoRR.csv')).iloc[:, 1:]
series_features = df_features.iloc[20:, :]['features']

lista_features = []
for element in series_features:
    lista_features.append(element.replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').split(','))

print('nº de features seleecionado:')
print(len(lista_features[0]))

lista_modelos = os.listdir(folder_modelos_con_CoRR)
lista_modelos = [modelo for modelo in lista_modelos if 'MLP_nfeats_40' in modelo]
lista_modelos.sort()

lista_datos_train = os.listdir(folder_modelos_con_CoRR)
lista_datos_train = [datos for datos in lista_datos_train if 'X_train_nfeats_40' in datos]
lista_datos_train.sort()

lista_datos_val = os.listdir(folder_modelos_con_CoRR)
lista_datos_val = [datos for datos in lista_datos_val if 'X_val_nfeats_40' in datos]
lista_datos_val.sort()

lista_df_datos_train = []
for datos in lista_datos_train:
    lista_df_datos_train.append(pd.read_csv(os.path.join(folder_modelos_con_CoRR, datos)).iloc[:, 1:])

lista_df_datos_val = []
for datos in lista_datos_val:
    lista_df_datos_val.append(pd.read_csv(os.path.join(folder_modelos_con_CoRR, datos)).iloc[:, 1:])

age_val = []
for val_df in lista_df_datos_val:
    age_val_fold = []
    for i in range(0, len(val_df), 1):
        age_val_fold.append(datos_todos[datos_todos['eTIV'] == val_df.loc[i, 'eTIV']]['Age'].values[0])
    age_val.append(age_val_fold)

pred_controles, pred_migr_cr, pred_migr_ep = [], [], []
lista_pred_val = []
for i in range(0, 10, 1):
    loaded_model = pickle.load(open(os.path.join(folder_modelos_con_CoRR, lista_modelos[i]), 'rb'))
    datos_train_norm, datos_val_norm, datos_control_sel_norm = outliers_y_normalizacion_3_entries(lista_df_datos_train[i], lista_df_datos_val[i], datos_control_sel)
    datos_train_norm, datos_val_norm, datos_migr_cr_norm = outliers_y_normalizacion_3_entries(lista_df_datos_train[i], lista_df_datos_val[i], datos_migr_cr)
    datos_train_norm, datos_val_norm, datos_migr_ep_norm = outliers_y_normalizacion_3_entries(lista_df_datos_train[i], lista_df_datos_val[i], datos_migr_ep)

    prediction_control = loaded_model.predict(datos_control_sel_norm[lista_features[i]].values)
    prediction_migr_cr = loaded_model.predict(datos_migr_cr_norm[lista_features[i]].values)
    prediction_migr_ep = loaded_model.predict(datos_migr_ep_norm[lista_features[i]].values)

    lista_pred_val.append(loaded_model.predict(datos_val_norm[lista_features[i]].values))

    pred_controles.append(prediction_control)
    pred_migr_cr.append(prediction_migr_cr)
    pred_migr_ep.append(prediction_migr_ep)

    print("Calculado split "+str(i))

# Corrección Brain Age bias
slopes, intercepts = [], []
for i in range(0, 10, 1):
    x = np.array(age_val[i]).reshape((-1, 1))
    y = np.array(lista_pred_val[i])
    model = LinearRegression().fit(x, y)
    intercepts.append(model.intercept_)
    slopes.append(model.coef_[0])

pred_controles_corrected, pred_migr_cr_corrected, pred_migr_ep_corrected = [], [], []
for i in range(0, 10, 1):
    pred_controles_corrected.append((pred_controles[i]-intercepts[i])/slopes[i])
    pred_migr_cr_corrected.append((pred_migr_cr[i]-intercepts[i])/slopes[i])
    pred_migr_ep_corrected.append((pred_migr_ep[i]-intercepts[i])/slopes[i])

pred_media_controles = sum(pred_controles)/10
pred_media_migr_cr = sum(pred_migr_cr)/10
pred_media_migr_ep = sum(pred_migr_ep)/10

pred_media_controles_cor = sum(pred_controles_corrected)/10
pred_media_migr_cr_cor = sum(pred_migr_cr_corrected)/10
pred_media_migr_ep_cor = sum(pred_migr_ep_corrected)/10

BAG_controles = pred_media_controles - edad_controles_sel
BAG_migr_cr = pred_media_migr_cr - edad_migr_cr
BAG_migr_ep = pred_media_migr_ep - edad_migr_ep

BAG_controles_cor = pred_media_controles_cor - edad_controles_sel
BAG_migr_cr_cor = pred_media_migr_cr_cor - edad_migr_cr
BAG_migr_ep_cor = pred_media_migr_ep_cor - edad_migr_ep

datos_cron_save['PRED'] = pred_media_migr_cr_cor
datos_cron_save['BAG'] = BAG_migr_cr_cor
datos_ep_save['PRED'] = pred_media_migr_ep_cor
datos_ep_save['BAG'] = BAG_migr_ep_cor

datos_cron_save['BAG_no_corr'] = BAG_migr_cr
datos_ep_save['BAG_no_corr'] = BAG_migr_ep

MAE_controles_todos, MAE_migr_cr_todos, MAE_migr_ep_todos = [], [], []
BAG_controles_todos, BAG_migr_cr_todos, BAG_migr_ep_todos = [], [], []
r_controles_todos, r_migr_cr_todos, r_migr_ep_todos = [], [], []

for i in range(0, 10, 1):
    MAE_controles_todos.append(mean_absolute_error(edad_controles_sel, pred_controles[i]))
    MAE_migr_cr_todos.append(mean_absolute_error(edad_migr_cr, pred_migr_cr[i]))
    MAE_migr_ep_todos.append(mean_absolute_error(edad_migr_ep, pred_migr_ep[i]))

    r_controles_todos.append(stats.pearsonr(edad_controles_sel, pred_controles[i])[0])
    r_migr_cr_todos.append(stats.pearsonr(edad_migr_cr, pred_migr_cr[i])[0])
    r_migr_ep_todos.append(stats.pearsonr(edad_migr_ep, pred_migr_ep[i])[0])

    BAG_controles_todos.append(edad_controles_sel - pred_controles[i])
    BAG_migr_cr_todos.append(edad_migr_cr - pred_migr_cr[i])
    BAG_migr_ep_todos.append(edad_migr_ep - pred_migr_ep[i])

print("[INFO] presento los valores de MAE para cada grupo (hecho con la edad media predicha)")
print("MAE Controles: "+str(mean_absolute_error(edad_controles_sel, pred_media_controles)))
print("r correlation:"+str(stats.pearsonr(edad_controles_sel, pred_media_controles)))
print("MAE migr cr: "+str(mean_absolute_error(edad_migr_cr, pred_media_migr_cr)))
print("r correlation migr cr:"+str(stats.pearsonr(edad_migr_cr, pred_media_migr_cr)))
print("MAE migr ep: "+str(mean_absolute_error(edad_migr_ep, pred_media_migr_ep)))
print("r correlation migr ep:"+str(stats.pearsonr(edad_migr_ep, pred_media_migr_ep)))

print("[INFO] presento los valores media de MAE para cada grupo")
print("MAE Controles: "+str(sum(MAE_controles_todos)/10))
print("MAE migr cr: "+str(sum(MAE_migr_cr_todos)/10))
print("MAE migr ep: "+str(sum(MAE_migr_ep_todos)/10)+'\n')

print("[INFO] presento los valores de std de MAE para cada grupo")
print("MAE Controles: "+str(np.array(MAE_controles_todos).std()))
print("MAE migr cr: "+str(np.array(MAE_migr_cr_todos).std()))
print("MAE migr ep: "+str(np.array(MAE_migr_ep_todos).std()))

print("[INFO] presento los valores media de MAE para cada grupo")
print("r Controles: "+str(sum(r_controles_todos)/10))
print("r migr cr: "+str(sum(r_migr_cr_todos)/10))
print("r migr ep: "+str(sum(r_migr_ep_todos)/10)+'\n')

print("[INFO] presento los valores de std de MAE para cada grupo")
print("r Controles: "+str(np.array(r_controles_todos).std()))
print("r migr cr: "+str(np.array(r_migr_cr_todos).std()))
print("r migr ep: "+str(np.array(r_migr_ep_todos).std()))

# BAG_controles = sum(BAG_controles_todos)/10
# BAG_migr_cr = sum(BAG_migr_cr_todos)/10
# BAG_migr_ep = sum(BAG_migr_ep_todos)/10

# 1.- Compruebo diferencias de sexo entre grupos xi cuadrado
print('Xi cuadrado para comprobar las diferencias entre sexos entre los grupos: ')
migr_cron_sex_count = np.unique(sex_migr_cr, return_index=False, return_inverse=False, return_counts=True)
migr_epi_sex_count = np.unique(sex_migr_ep, return_index=False, return_inverse=False, return_counts=True)

# Estaba mal el sexo en algunos pacientes en mi excel => refiero a excel de adquisiciones
migr_control_sex_count = np.unique(sex_control_sel, return_index=False, return_inverse=False, return_counts=True)
obs = np.array([migr_cron_sex_count[1], migr_epi_sex_count[1], migr_control_sex_count[1]])
print(chi2_contingency(obs))
print('Si p-val > 0.05, no hay diferencias entre sexos'+'\n')

# 2.- Compruebo que las edades tengan una distribución normal test de Smirnov-Kolmogorov
print('Pruebo normalidad de la edad de los grupos (Kolmogorov-Smirnov test).')
años_migraña_episódica_standard = (edad_migr_ep-np.mean(edad_migr_ep))/np.std(edad_migr_ep)
print(kstest(años_migraña_episódica_standard, 'norm'))
años_migraña_crónica_standard = (edad_migr_cr-np.mean(edad_migr_cr))/np.std(edad_migr_cr)
print(kstest(años_migraña_crónica_standard, 'norm'))
años_controles_standard = (edad_controles_sel-np.mean(edad_controles_sel))/np.std(edad_controles_sel)
print(kstest(años_controles_standard, 'norm'))
print('Si p-valor es < 0.05 es que no son normales'+'\n')

# 3.- Compruebo que las edades tengan una varianza homocedastica test de Levene
print('Pruebo la igualdad de varianzas entre los grupos de edad.')
print(levene(edad_migr_ep, edad_migr_cr, edad_controles_sel))
print('Prueba de Levene para igualdad de Varianzas pval < 0.05 varianzas distintas'+'\n')

# Veo diferencias entre migr cr y migr ep y controles; ANOVA y t-test?
print('Hago ANOVA ya que sale que no hay diferencias de varianza y las distribuciones son normales')
print(f_oneway(edad_migr_ep, edad_migr_cr, edad_controles_sel))
print('Si pval < 0.05 son distintos'+'\n')

print("[INFO] He comprobado que los grupos por edad y por sexo son comparables")
print("[INFO] Compruebo si puedo hacer el test con el BAG"+'\n')

# 1.- Compruebo que los BAG tengan una distribución normal test de Smirnov-Kolmogorov
print('Pruebo normalidad de los BAG (Kolmogorov-Smirnov test).')
print('controles')
BAG_controles_standard = (BAG_controles-np.mean(BAG_controles))/np.std(BAG_controles)
print(kstest(BAG_controles_standard, 'norm'))
print('migr cr')
BAG_migr_cr_standard = (BAG_migr_cr-np.mean(BAG_migr_cr))/np.std(BAG_migr_cr)
print(kstest(años_migraña_crónica_standard, 'norm'))
print('migr ep')
BAG_migr_ep_standard = (BAG_migr_ep-np.mean(BAG_migr_ep))/np.std(BAG_migr_ep)
print(kstest(BAG_migr_ep_standard, 'norm'))
print('Si p-valor es < 0.05 es que no son normales'+'\n')

# 2.- Compruebo que los BAG tengan una varianza homocedastica test de Levene
print('Pruebo la igualdad de varianzas entre los grupos de BAG.')
print(levene(BAG_migr_ep, BAG_migr_cr, BAG_controles))
print('Prueba de Levene para igualdad de Varianzas pval < 0.05 varianzas distintas'+'\n')

# Veo diferencias entre migr cr y migr ep y controles; ANOVA y t-test?
print('Hago ANCOVA ya que sale que no hay diferencias de varianza y las distribuciones son normales BAG')

etiv_migr_ep = datos_migr_ep['eTIV'].values
etiv_migr_cr = datos_migr_cr['eTIV'].values
etiv_control_sel = datos_control_sel['eTIV'].values
ages = np.concatenate((edad_controles_sel, edad_migr_ep))
ages = np.concatenate((ages, edad_migr_cr))
sexos = np.concatenate((sex_control_sel, sex_migr_ep))
sexos = np.concatenate((sexos, sex_migr_cr))
sexos_cat = []
for value in sexos:
    if value =='F':
        sexos_cat.append(1)
    else:
        sexos_cat.append(0)
BAG = np.concatenate((BAG_controles, BAG_migr_ep))
BAG = np.concatenate((BAG, BAG_migr_cr))
etivs = np.concatenate((etiv_control_sel, etiv_migr_ep))
etivs = np.concatenate((etivs, etiv_migr_cr))
type_bag = np.concatenate((np.repeat('controles', len(BAG_controles)), np.repeat('migr_ep', len(BAG_migr_ep))))
type_bag = np.hstack((type_bag, np.repeat('migr_cr', len(BAG_migr_cr))))

df_ancova = {'edad_real': ages.tolist(), 'sexos': sexos_cat, 'etivs': etivs.tolist(), 'BAG': BAG.tolist(), 'type': type_bag.tolist()}
df_ancova = pd.DataFrame.from_dict(df_ancova)

print(ancova(data=df_ancova, dv='BAG', covar=['edad_real', 'etivs', 'sexos'], between='type'))

print('Si pval < 0.05 son distintos'+'\n')

# t-test controles migr ep
print("[INFO] TEST Controles - Migraña episódica #")
print('Pruebo normalidad:')
print('YA PROBADO ANTES TODOS SON NORMALES\n')

print('Pruebo la igualdad de varianzas entre BAG controles y BAG migraña episódica:')
print(levene(BAG_controles, BAG_migr_ep))
print('Prueba de Levene para igualdad de Varianzas pval < 0.05 varianzas distintas'+'\n')

etiv_migr_ep = datos_migr_ep['eTIV'].values
etiv_control_sel = datos_control_sel['eTIV'].values
ages = np.concatenate((edad_controles_sel, edad_migr_ep))
sexos = np.concatenate((sex_control_sel, sex_migr_ep))
sexos_cat = []
for value in sexos:
    if value =='F':
        sexos_cat.append(1)
    else:
        sexos_cat.append(0)
BAG = np.concatenate((BAG_controles, BAG_migr_ep))
etivs = np.concatenate((etiv_control_sel, etiv_migr_ep))
type_bag = np.concatenate((np.repeat('controles', len(BAG_controles)), np.repeat('migr_ep', len(BAG_migr_ep))))

df_ancova = {'edad_real': ages.tolist(), 'sexos': sexos_cat, 'etivs': etivs.tolist(), 'BAG': BAG.tolist(), 'type': type_bag.tolist()}
df_ancova = pd.DataFrame.from_dict(df_ancova)

print(ancova(data=df_ancova, dv='BAG', covar=['edad_real', 'etivs', 'sexos'], between='type'))

# t-test controles migr cr
print("[INFO] T-TEST Controles - Migraña crónica #")
print('Pruebo normalidad:')
print('YA PROBADO ANTES TODOS SON NORMALES\n')

print('Pruebo la igualdad de varianzas entre BAG controles y BAG migraña crónica:')
print(levene(BAG_controles, BAG_migr_cr))
print('Prueba de Levene para igualdad de Varianzas pval < 0.05 varianzas distintas'+'\n')

etiv_migr_cr = datos_migr_cr['eTIV'].values
etiv_control_sel = datos_control_sel['eTIV'].values
ages = np.concatenate((edad_controles_sel, edad_migr_cr))
sexos = np.concatenate((sex_control_sel, sex_migr_cr))
sexos_cat = []
for value in sexos:
    if value =='F':
        sexos_cat.append(1)
    else:
        sexos_cat.append(0)
BAG = np.concatenate((BAG_controles, BAG_migr_cr))
etivs = np.concatenate((etiv_control_sel, etiv_migr_cr))
type_bag = np.concatenate((np.repeat('controles', len(BAG_controles)), np.repeat('migr_cr', len(BAG_migr_cr))))

df_ancova = {'edad_real': ages.tolist(), 'sexos': sexos_cat, 'etivs': etivs.tolist(), 'BAG': BAG.tolist(), 'type': type_bag.tolist()}
df_ancova = pd.DataFrame.from_dict(df_ancova)

print(ancova(data=df_ancova, dv='BAG', covar=['edad_real', 'etivs', 'sexos'], between='type'))

# t-tes t migr cr migr ep
print("[INFO] T-TEST Migraña episódica - Migraña crónica #")
print('Pruebo normalidad:')
print('YA PROBADO ANTES TODOS SON NORMALES\n')

print('Pruebo la igualdad de varianzas entre BAG migraña episódica y BAG migraña crónica:')
print(levene(BAG_migr_ep, BAG_migr_cr))
print('Prueba de Levene para igualdad de Varianzas pval < 0.05 varianzas distintas'+'\n')

etiv_migr_ep = datos_migr_ep['eTIV'].values
etiv_migr_cr = datos_migr_cr['eTIV'].values
etiv_control_sel = datos_control_sel['eTIV'].values
ages = np.concatenate((edad_migr_ep, edad_migr_cr))
sexos = np.concatenate((sex_migr_ep, sex_migr_cr))
sexos_cat = []
for value in sexos:
    if value =='F':
        sexos_cat.append(1)
    else:
        sexos_cat.append(0)
BAG = np.concatenate((BAG_migr_ep, BAG_migr_cr))
etivs = np.concatenate((etiv_migr_ep, etiv_migr_cr))
type_bag = np.concatenate((np.repeat('migr_ep', len(BAG_migr_ep)), np.repeat('migr_cr', len(BAG_migr_cr))))
df_ancova = {'edad_real': ages.tolist(), 'sexos': sexos_cat, 'etivs': etivs.tolist(), 'BAG': BAG.tolist(), 'type': type_bag.tolist()}
df_ancova = pd.DataFrame.from_dict(df_ancova)

print(ancova(data=df_ancova, dv='BAG', covar=['edad_real', 'etivs', 'sexos'], between='type'))

# t-tes t migr cr migr ep
print("[INFO] T-TEST Migraña - controles #")
print('Pruebo normalidad:')
print('YA PROBADO ANTES TODOS SON NORMALES\n')
print('migraña todo')
BAG_migr = np.concatenate((BAG_migr_ep, BAG_migr_cr))
BAG_migr_norm = (BAG_migr-np.mean(BAG_migr))/np.std(BAG_migr)
print(kstest(BAG_migr_norm, 'norm'))
print('Pruebo la igualdad de varianzas entre BAG migraña todos y BAG controles:')
print(levene(BAG_controles, BAG_migr))
print('Prueba de Levene para igualdad de Varianzas pval < 0.05 varianzas distintas'+'\n')

etiv_migr_ep = datos_migr_ep['eTIV'].values
etiv_migr_cr = datos_migr_cr['eTIV'].values
etiv_control_sel = datos_control_sel['eTIV'].values
ages = np.concatenate((edad_migr_ep, edad_migr_cr))
ages = np.concatenate((ages, edad_controles_sel))
sexos = np.concatenate((sex_migr_ep, sex_migr_cr))
sexos = np.concatenate((sexos, sex_control_sel))
sexos_cat = []
for value in sexos:
    if value =='F':
        sexos_cat.append(1)
    else:
        sexos_cat.append(0)
BAG = np.concatenate((BAG_migr, BAG_controles))
etivs = np.concatenate((etiv_migr_ep, etiv_migr_cr))
etivs = np.concatenate((etivs, etiv_control_sel))
type_bag = np.concatenate((np.repeat('migr', len(BAG_migr_ep)), np.repeat('migr', len(BAG_migr_cr))))
type_bag = np.concatenate((type_bag, np.repeat('controles', len(BAG_controles))))
df_ancova = {'edad_real': ages.tolist(), 'sexos': sexos_cat, 'etivs': etivs.tolist(), 'BAG': BAG.tolist(), 'type': type_bag.tolist()}
df_ancova = pd.DataFrame.from_dict(df_ancova)

print(welch_anova(data=df_ancova, dv='BAG', between='type'))
print(ancova(data=df_ancova, dv='BAG', covar=['edad_real', 'etivs', 'sexos'], between='type'))

# Colors taken from Dark2 palette in RColorBrewer R library
# COLOR_SCALE = ["#789837", "#5481A6", "#FEA32A"]

# plot edad predicha vs edad real en los tres grupos
# controles
edades_real = np.concatenate((edad_migr_ep, edad_migr_cr))
edades_real = np.concatenate((edades_real, edad_controles_sel))
edades_predichas = np.concatenate((pred_media_migr_ep, pred_media_migr_cr))
edades_predichas = np.concatenate((edades_predichas, pred_media_controles))
type_bag = np.concatenate((np.repeat('migr_ep', len(BAG_migr_ep)), np.repeat('migr_cr', len(BAG_migr_cr))))
type_bag = np.hstack((type_bag, np.repeat('controles', len(BAG_controles))))

df_scatter = {'edad_real': edades_real.tolist(), 'edad_predicha': edades_predichas.tolist(), 'type': type_bag.tolist()}
df_scatter = pd.DataFrame.from_dict(df_scatter)
# sns.regplot(x="edad_real", y="edad_predicha", ci=None, data=df_scatter)

# sns.scatterplot(data=df_scatter, x="edad_real", y="edad_predicha", hue="type")

x=df_scatter[df_scatter['type'] =='controles']['edad_real'].values
y=df_scatter[df_scatter['type'] =='controles']['edad_predicha'].values
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#789837', alpha=1, marker="^", label="Controles")
plt.plot(x, poly1d_fn(x), color='#789837')

x=df_scatter[df_scatter['type'] =='migr_ep']['edad_real'].values
y=df_scatter[df_scatter['type'] =='migr_ep']['edad_predicha'].values
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#5481A6', alpha=1, marker="*", label="Migraña episódica")
plt.plot(x, poly1d_fn(x), color='#5481A6')

x=df_scatter[df_scatter['type'] =='migr_cr']['edad_real'].values
y=df_scatter[df_scatter['type'] =='migr_cr']['edad_predicha'].values
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#FEA32A', alpha=1, marker="d", label="Migraña crónica")
plt.plot(x, poly1d_fn(x), color='#FEA32A')

X_plot = np.linspace(18, 60, 42)
Y_plot = np.linspace(18, 60, 42)
plt.legend(loc="upper left")
plt.plot(X_plot, Y_plot, '--k', linewidth=0.7)
plt.title('Resultado Brain Age', fontweight='bold')
plt.ylabel('Edad Predicha', fontweight='bold')
plt.xlabel('Edad Real', fontweight='bold')
plt.show()

# Controles
df_scatter = {'edad_real': edad_controles_sel.tolist(), 'edad_predicha': pred_media_controles.tolist()}
df_scatter = pd.DataFrame.from_dict(df_scatter)
sns.regplot(x="edad_real", y="edad_predicha", ci=None, data=df_scatter)
X_plot = np.linspace(18, 60, 42)
Y_plot = np.linspace(18, 60, 42)
plt.plot(X_plot, Y_plot, color='r')
plt.ylabel('Edad Predicha', fontweight='bold')
plt.xlabel('Edad Real', fontweight='bold')
plt.title('Controles', fontweight='bold')
plt.show()

# migraña ep
df_scatter = {'edad_real': edad_migr_ep.tolist(), 'edad_predicha': pred_media_migr_ep.tolist()}
df_scatter = pd.DataFrame.from_dict(df_scatter)
sns.regplot(x="edad_real", y="edad_predicha", ci=None, data=df_scatter)
X_plot = np.linspace(18, 60, 42)
Y_plot = np.linspace(18, 60, 42)
plt.plot(X_plot, Y_plot, color='r')
plt.ylabel('Edad Predicha', fontweight='bold')
plt.xlabel('Edad Real', fontweight='bold')
plt.title('Migraña Episódica', fontweight='bold')
plt.show()

# migraña cronica
df_scatter = {'edad_real': edad_migr_cr.tolist(), 'edad_predicha': pred_media_migr_cr.tolist()}
df_scatter = pd.DataFrame.from_dict(df_scatter)
sns.regplot(x="edad_real", y="edad_predicha", ci=None, data=df_scatter)
X_plot = np.linspace(18, 60, 42)
Y_plot = np.linspace(18, 60, 42)
plt.plot(X_plot, Y_plot, color='r')
plt.ylabel('Edad Predicha', fontweight='bold')
plt.xlabel('Edad Real', fontweight='bold')
plt.title('Migraña Crónica', fontweight='bold')
plt.show()

# Hago el violin plot
BAG = np.concatenate((BAG_controles, BAG_migr_ep))
BAG = np.concatenate((BAG, BAG_migr_cr))
type_bag = np.concatenate((np.repeat('controles', len(BAG_controles)), np.repeat('migr_ep', len(BAG_migr_ep))))
type_bag = np.hstack((type_bag, np.repeat('migr_cr', len(BAG_migr_cr))))

d = {'BAG': BAG.tolist(), 'type': type_bag.tolist()}
dataframe_violin_plot = pd.DataFrame(data=d)
sns.violinplot(data=dataframe_violin_plot, x="type", y="BAG")
# plt.plot([18, 60], [18, 60], ls="--", c=".3")
plt.title('distribución BAG para cada uno de los grupos', fontweight='bold')
plt.ylabel('BAG (años)', fontweight='bold')
plt.xlabel('Grupo', fontweight='bold')
plt.ylim(-35, 35)
plt.show()

migr_cron_clin = config_parser.get("DATOS_CLIN", "migr_cron_clin")
migr_ep_clin = config_parser.get("DATOS_CLIN", "migr_ep_clin")

migr_cron_clin = pd.read_excel(migr_cron_clin)
migr_ep_clin = pd.read_excel(migr_ep_clin)

migr_cron_clin = migr_cron_clin.rename(columns={"ID Excel": "ID"})
migr_ep_clin = migr_ep_clin.rename(columns={"ID Excel": "ID"})

cron_merged = pd.merge(datos_cron_save, migr_cron_clin, on="ID")
ep_merged = pd.merge(datos_ep_save, migr_ep_clin, on="ID")

# Reset index after drop
cron_merged = cron_merged.dropna().reset_index(drop=True)
ep_merged = ep_merged.dropna().reset_index(drop=True)

# plot con
x = cron_merged['tiempo_mig (años)'].values
y = cron_merged['BAG'].values
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#339989', alpha=1, marker="o")
corr_cron = stats.pearsonr(x, y)
plt.plot(x, poly1d_fn(x), color='#339989', label='MC r:{:.2f}; p-val:{:.2f}'.format(corr_cron[0], corr_cron[1]))

# plot con
x = ep_merged['tiempo_mig (años)'].values
y = ep_merged['BAG'].values
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#003366', alpha=1, marker="o")
corr_ep = stats.pearsonr(x, y)
plt.plot(x, poly1d_fn(x), color='#003366', label='ME r:{:.2f}; p-val:{:.2f}'.format(corr_ep[0], corr_ep[1]))

x = np.array(ep_merged['tiempo_mig (años)'].values.tolist()+cron_merged['tiempo_mig (años)'].values.tolist())
y = np.array(ep_merged['BAG'].values.tolist()+cron_merged['BAG'].values.tolist())
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
corr_all = stats.pearsonr(x, y)
plt.plot(x, poly1d_fn(x), color='#69140E', label='All r:{:.2f}; p-val:{:.2f}'.format(corr_all[0], corr_all[1]))

# plt.title('BrainAGE frente a los años de duración de la migraña', fontweight='bold')
plt.xlabel('Migraine duration (years)', fontweight='bold')
plt.ylabel('BrainAGE (years)', fontweight='bold')
plt.legend(fontsize=9)
plt.show()

# plot con
x = cron_merged['freq_cef (veces/mes)'].values
y = cron_merged['BAG'].values
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#339989', alpha=1, marker="o")
corr_cron = stats.pearsonr(x, y)
plt.plot(x, poly1d_fn(x), color='#339989', label='MC r:{:.2f}; p-val:{:.2f}'.format(corr_cron[0], corr_cron[1]))

x = ep_merged['freq_cef (veces/mes)'].values
y = ep_merged['BAG'].values
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#003366', alpha=1, marker="o")
corr_ep = stats.pearsonr(x, y)
plt.plot(x, poly1d_fn(x), color='#003366', label='ME r:{:.2f}; p-val:{:.2f}'.format(corr_ep[0], corr_ep[1]))
# plt.title('variación de BrainAGE frente a la frecuencia de migrañas', fontweight='bold')

x = np.array(ep_merged['freq_cef (veces/mes)'].values.tolist()+cron_merged['freq_cef (veces/mes)'].values.tolist())
y = np.array(ep_merged['BAG'].values.tolist()+cron_merged['BAG'].values.tolist())
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
corr_all = stats.pearsonr(x, y)
plt.plot(x, poly1d_fn(x), color='#69140E', label='All r:{:.2f}; p-val:{:.2f}'.format(corr_all[0], corr_all[1]))

plt.xlabel('Headache frequency (bouts/month)', fontweight='bold')
plt.ylabel('BrainAGE (years)', fontweight='bold')
plt.legend(fontsize=9)
plt.show()

# plot con
x = cron_merged['freq_mig (veces/mes)'].values
y = cron_merged['BAG'].values
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#339989', alpha=1, marker="o")
corr_cron = stats.pearsonr(x, y)
plt.plot(x, poly1d_fn(x), color='#339989', label='MC r:{:.2f}; p-val:{:.2f}'.format(corr_cron[0], corr_cron[1]))

x = ep_merged['freq_mig (veces/mes)'].values
y = ep_merged['BAG'].values
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#003366', alpha=1, marker="o")
corr_ep = stats.pearsonr(x, y)
plt.plot(x, poly1d_fn(x), color='#003366', label='ME r:{:.2f}; p-val:{:.2f}'.format(corr_ep[0], corr_ep[1]))
# plt.title('variación de BrainAGE frente a la frecuencia de cefaleas',  fontweight='bold')

x = np.array(ep_merged['freq_mig (veces/mes)'].values.tolist()+cron_merged['freq_mig (veces/mes)'].values.tolist())
y = np.array(ep_merged['BAG'].values.tolist()+cron_merged['BAG'].values.tolist())
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
corr_all = stats.pearsonr(x, y)
plt.plot(x, poly1d_fn(x), color='#69140E', label='All r:{:.2f}; p-val:{:.2f}'.format(corr_all[0], corr_all[1]))

plt.xlabel('Migraine frequency (bouts/month)', fontweight='bold')
plt.ylabel('BrainAGE (years)', fontweight='bold')
plt.legend(fontsize=9)
plt.show()

x = cron_merged['tiempo_mig_cro (meses)'].values
y = cron_merged['BAG'].values
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#339989', alpha=1, marker="o")
corr_cron = stats.pearsonr(x, y)
plt.plot(x, poly1d_fn(x), color='#339989', label='MC r:{:.2f}; p-val:{:.2f}'.format(corr_cron[0], corr_cron[1]))

plt.xlabel('Chronic migraine duration (months)', fontweight='bold')
plt.ylabel('BrainAGE (years)', fontweight='bold')
plt.legend(fontsize=9)
plt.show()

# REPITO ANALISIS CON AÑOS MIGRAÑA COMO COVARIABLE

print('Pruebo la igualdad de varianzas entre BAG controles y BAG migraña episódica (2):')
print(levene(BAG_controles, BAG_migr_ep))
print('Prueba de Levene para igualdad de Varianzas pval < 0.05 varianzas distintas'+'\n')

ep_merged = pd.merge(datos_migr_ep, ep_merged, on="ID", how='right')

etiv_migr_ep = ep_merged['eTIV'].values
etiv_control_sel = datos_control_sel['eTIV'].values
ages = np.concatenate((edad_controles_sel, ep_merged['Age_x'].values))
sexos = np.concatenate((sex_control_sel, ep_merged['M/F_x'].values))
anos_mig = np.concatenate((np.repeat(0, len(BAG_controles)), ep_merged['tiempo_mig (años)'].values))
sexos_cat = []
for value in sexos:
    if value =='F':
        sexos_cat.append(1)
    else:
        sexos_cat.append(0)
BAG = np.concatenate((BAG_controles, ep_merged['BAG'].values))
etivs = np.concatenate((etiv_control_sel, etiv_migr_ep))
type_bag = np.concatenate((np.repeat('controles', len(BAG_controles)), np.repeat('migr_ep', len(ep_merged['BAG'].values))))

df_ancova = {'edad_real': ages.tolist(), 'sexos': sexos_cat, 'etivs': etivs.tolist(), 'BAG': BAG.tolist(), 'type': type_bag.tolist(), 'años_mig': anos_mig}
df_ancova = pd.DataFrame.from_dict(df_ancova)

print(ancova(data=df_ancova, dv='BAG', covar=['edad_real', 'etivs', 'sexos', 'años_mig'], between='type'))

# REPITO ANALISIS CON AÑOS MIGRAÑA COMO COVARIABLE

print('Pruebo la igualdad de varianzas entre BAG controles y BAG migraña crónica (2):')
print(levene(BAG_controles, BAG_migr_cr))
print('Prueba de Levene para igualdad de Varianzas pval < 0.05 varianzas distintas'+'\n')

cron_merged = pd.merge(datos_migr_cr, cron_merged, on="ID", how='right')

etiv_migr_ep = cron_merged['eTIV'].values
etiv_control_sel = datos_control_sel['eTIV'].values
ages = np.concatenate((edad_controles_sel, cron_merged['Age_x'].values))
sexos = np.concatenate((sex_control_sel, cron_merged['M/F_x'].values))
anos_mig = np.concatenate((np.repeat(0, len(BAG_controles)), cron_merged['tiempo_mig (años)'].values))
sexos_cat = []
for value in sexos:
    if value =='F':
        sexos_cat.append(1)
    else:
        sexos_cat.append(0)
BAG = np.concatenate((BAG_controles, cron_merged['BAG'].values))
etivs = np.concatenate((etiv_control_sel, etiv_migr_ep))
type_bag = np.concatenate((np.repeat('controles', len(BAG_controles)), np.repeat('migr_cr', len(cron_merged['BAG'].values))))

df_ancova = {'edad_real': ages.tolist(), 'sexos': sexos_cat, 'etivs': etivs.tolist(), 'BAG': BAG.tolist(), 'type': type_bag.tolist(), 'años_mig': anos_mig}
df_ancova = pd.DataFrame.from_dict(df_ancova)

print(ancova(data=df_ancova, dv='BAG', covar=['edad_real', 'etivs', 'sexos', 'años_mig'], between='type'))

# Busco diferencias estadísticas entre la duración de migraña entre ME y MC
print('Duracion de migraña en años migraña episodica media y stdev:')
print(np.mean(ep_merged['tiempo_mig (años)'].values))
print(np.std(ep_merged['tiempo_mig (años)'].values))
print('\n')

print('Duracion de migraña en años migraña episodica media y stdev:')
print(np.mean(cron_merged['tiempo_mig (años)'].values))
print(np.std(cron_merged['tiempo_mig (años)'].values))
print('\n')

# Condicioens
# 1.- Normalidad; si son normales
duracion_migraña_episodica_standard = (ep_merged['tiempo_mig (años)'].values -np.mean(ep_merged['tiempo_mig (años)'].values))/np.std(ep_merged['tiempo_mig (años)'].values)
duracion_migraña_cronica_standard = (cron_merged['tiempo_mig (años)'].values -np.mean(cron_merged['tiempo_mig (años)'].values))/np.std(cron_merged['tiempo_mig (años)'].values)
print('Check si hay normalidad en diración de migraña episodica y crónica:')
print(kstest(duracion_migraña_episodica_standard, 'norm'))
print(kstest(duracion_migraña_cronica_standard, 'norm'))
print('\n')

# 2.- igualdad de varianzas; sí lo es
print('Check si la variaza es comparable entre los grupos')
print(levene(ep_merged['tiempo_mig (años)'].values, cron_merged['tiempo_mig (años)'].values))
print('\n')

# 3.- Hago el t-test
print(stats.ttest_ind(a=ep_merged['tiempo_mig (años)'].values, b=cron_merged['tiempo_mig (años)'].values, equal_var=True))

# Busco diferencias estadísticas entre la freq de cefalea de ME y MC
print('freq_cef en migraña episodica media y stdev:')
print(np.mean(ep_merged['freq_cef (veces/mes)'].values))
print(np.std(ep_merged['freq_cef (veces/mes)'].values))
print('\n')

print('freq_cef en migraña episodica media y stdev:')
print(np.mean(cron_merged['freq_cef (veces/mes)'].values))
print(np.std(cron_merged['freq_cef (veces/mes)'].values))
print('\n')

# Condiciones
# 1.- Normalidad; NO NORMAL
freq_cef_episodica_standard = (ep_merged['freq_cef (veces/mes)'].values -np.mean(ep_merged['freq_cef (veces/mes)'].values))/np.std(ep_merged['freq_cef (veces/mes)'].values)
freq_cef_cronica_standard = (cron_merged['freq_cef (veces/mes)'].values -np.mean(cron_merged['freq_cef (veces/mes)'].values))/np.std(cron_merged['freq_cef (veces/mes)'].values)
print('Check si hay normalidad en freq_cef episodica y crónica:')
print(kstest(freq_cef_episodica_standard, 'norm'))
print(kstest(freq_cef_cronica_standard, 'norm'))
print('\n')

# 2.- igualdad de varianzas; NO VARIANZA
print('Check si la variaza es comparable entre los grupos')
print(levene(ep_merged['freq_cef (veces/mes)'].values, cron_merged['freq_cef (veces/mes)'].values))
print('\n')

# 3.- Hago el t-test
print(stats.mannwhitneyu(ep_merged['freq_cef (veces/mes)'].values, cron_merged['freq_cef (veces/mes)'].values, method="auto"))

# Busco diferencias estadísticas entre la freq de migraña de ME y MC
print('freq_mig en migraña episodica media y stdev:')
print(np.mean(ep_merged['freq_mig (veces/mes)'].values))
print(np.std(ep_merged['freq_mig (veces/mes)'].values))
print('\n')

print('freq_mig en migraña episodica media y stdev:')
print(np.mean(cron_merged['freq_mig (veces/mes)'].values))
print(np.std(cron_merged['freq_mig (veces/mes)'].values))
print('\n')

# Condicioens
# 1.- Normalidad
freq_mig_episodica_standard = (ep_merged['freq_mig (veces/mes)'].values -np.mean(ep_merged['freq_mig (veces/mes)'].values))/np.std(ep_merged['freq_mig (veces/mes)'].values)
freq_mig_cronica_standard = (cron_merged['freq_mig (veces/mes)'].values -np.mean(cron_merged['freq_mig (veces/mes)'].values))/np.std(cron_merged['freq_mig (veces/mes)'].values)
print('Check si hay normalidad en freq_mig episodica y crónica:')
print(kstest(freq_mig_episodica_standard, 'norm'))
print(kstest(freq_mig_cronica_standard, 'norm'))
print('\n')

# 2.- igualdad de varianzas
print('Check si la variaza es comparable entre los grupos')
print(levene(ep_merged['freq_mig (veces/mes)'].values, cron_merged['freq_mig (veces/mes)'].values))
print('\n')

# Hago el t-test
print(stats.mannwhitneyu(ep_merged['freq_mig (veces/mes)'].values, cron_merged['freq_mig (veces/mes)'].values, method="auto"))

# HAGO PLOTS DE LAS REGIONES IDENTIFICADAS COMO DIFERENTES EN LOS SHAP VALUES
# GrayVol_lh_superiorfrontal
# GrayVol_lh_lateralorbitofrontal

# plot con
# x = cron_merged['freq_mig (veces/mes)'].values
# y = cron_merged['GrayVol_lh_superiorfrontal'].values
# coef = np.polyfit(x, y, 1)
# poly1d_fn = np.poly1d(coef)
# plt.scatter(x, y, color='#339989', alpha=1, marker="o")
# corr_cron = stats.pearsonr(x, y)
# plt.plot(x, poly1d_fn(x), color='#339989', label='MC r:{:.2f}; p-val:{:.2f}'.format(corr_cron[0], corr_cron[1]))


x = np.array(ep_merged['freq_cef (veces/mes)'].values.tolist()+cron_merged['freq_cef (veces/mes)'].values.tolist())
y = np.array(ep_merged['GrayVol_lh_lateralorbitofrontal'].values.tolist()+cron_merged['GrayVol_lh_lateralorbitofrontal'].values.tolist())
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#003366', alpha=1, marker="o")
corr_ep = stats.pearsonr(x, y)
plt.plot(x, poly1d_fn(x), color='#003366', label='ME r:{:.2f}; p-val:{:.2f}'.format(corr_ep[0], corr_ep[1]))
# plt.title('variación de BrainAGE frente a la frecuencia de cefaleas',  fontweight='bold')
plt.xlabel('freq_cef (veces/mes)', fontweight='bold')
plt.ylabel('GrayVol_lh_superiorfrontal', fontweight='bold')
plt.legend(fontsize=9)
plt.show()


key_features = ['Mean_rh_wg_pct_precentral', 'normMax_Right-Caudate', 'ThickAvg_lh_parstriangularis', 'ThickAvg_rh_parsopercularis',
 'normStdDev_Left-Cerebellum-Cortex', 'normMax_3rd-Ventricle', 'GrayVol_lh_lateralorbitofrontal', 'normMaxwm-lh-parsorbitalis',
 'GrayVol_lh_superiorfrontal', 'Mean_lh_wg_pct_paracentral', 'Mean_lh_wg_pct_precentral', 'normMax_Right-Pallidum',
 'Volume_mm3_Left-Putamen', 'normStdDev_Left-Caudate', 'normMax_Left-Pallidum', 'ThickAvg_lh_insula']

key_features_morpho = ['GrayVol_lh_lateralorbitofrontal', 'MeanCurv_rh_transversetemporal', 'ThickAvg_lh_parsopercularis',
                       'ThickAvg_lh_precentral', 'Volume_mm3_Right-Putamen', 'GrayVol_rh_supramarginal', 'GrayVol_rh_superiorfrontal',
                       'TotalGrayVol', 'ThickAvg_rh_precentral', 'lh_MeanThickness', 'GrayVol_rh_insula', 'FoldInd_rh_rostralmiddlefrontal',
                       'GrayVol_lh_superiorfrontal', 'SubCortGrayVol', 'GrayVol_rh_middletemporal',
                       'ThickStd_rh_superiorfrontal', 'Volume_mm3_Left-Caudate', 'BAG']


for feature in key_features_morpho:
    x = np.array((ep_merged['freq_cef (veces/mes)']).tolist()+(cron_merged['freq_cef (veces/mes)']).tolist())
    # x = np.array(ep_merged['freq_cef (veces/mes)'].values.tolist()+cron_merged['freq_cef (veces/mes)'].values.tolist())
    y = np.array(ep_merged[feature].values.tolist()+cron_merged[feature].values.tolist())
    coef = np.polyfit(x, y, 1)
    poly1d_fn = np.poly1d(coef)
    plt.scatter(x, y, color='#003366', alpha=1, marker="o")
    corr_ep = stats.pearsonr(x, y)
    plt.plot(x, poly1d_fn(x), color='#003366', label='ME r:{:.2f}; p-val:{:.2f}'.format(corr_ep[0], corr_ep[1]))
    # plt.title('variación de BrainAGE frente a la frecuencia de cefaleas',  fontweight='bold')
    plt.xlabel('freq_cef', fontweight='bold')
    plt.ylabel(feature, fontweight='bold')
    plt.legend(fontsize=9)
    plt.show()
    print('Pausa')

print('fin')
print('fin')
print('fin')
