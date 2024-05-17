from Utils import *

import os
import configparser

from sklearn.metrics import mean_absolute_error

from pingouin import ancova

from scipy.stats import f_oneway
from scipy.stats import kstest, levene, chi2_contingency
from scipy import stats
import pingouin as pg

import seaborn as sns

def load_list(filename):
    with open(filename, 'rb') as file:
        loaded_list = pickle.load(file)
    return loaded_list

# config parser llamo al archivo de configuración
config_parser = configparser.ConfigParser(allow_no_value=True)
bindir = os.path.abspath(os.path.dirname(__file__))
config_parser.read(bindir + "/cfg.cnf")

folderApplication = config_parser.get("DATA", "DataAplication")
folderCreation = config_parser.get("DATA", "DataCreation")

# Change the model to use here
model = 'MLP40_Combined'
model_type = model[6:]
folder_models = os.path.join(config_parser.get("MODELS", "modelsFolder"), model)

print('[## SANITY CHECK ##] Selected model:')
print(model, end='\n\n')

# Load model and features; Best model MLP40 features in validation.
features_selected = pd.read_csv(config_parser.get("DATA", "FeaturesSelected"+model_type)).iloc[:, 1:]
series_features = features_selected.iloc[20:, :]['features']
features_list = [element.replace('[', '').replace(']', '').replace('\'', '').replace(' ', '').split(',') for element in series_features]

print('[## SANITY CHECK ##] Selected number of features:')
n_features = len(features_list[0])
print(n_features, end='\n\n')

print('[## INFO ##] Loading models and data...')

print('[## INFO ##] Sex selection of the  patients  to evaluate...', end='\n\n')
sex = 'Males' # 'Females'; 'Males'; "All"
print('[## INFO ##] Selected: '+sex, end='\n\n')

covar_EM = load_list(os.path.join(folderApplication, 'Demographics_and_Clinical', 'covariates_EM.pkl'))
covar_CM = load_list(os.path.join(folderApplication, 'Demographics_and_Clinical', 'covariates_CM.pkl'))
covar_HC = load_list(os.path.join(folderApplication, 'Demographics_and_Clinical', 'covariates_HC.pkl'))

# load normalized controls, episodic migraine data and crhonic migraine data
HC_data = load_list(os.path.join(folderApplication, 'DATA', 'HC_data_norm_' + model_type.lower() + '.pkl'))
EM_data = load_list(os.path.join(folderApplication, 'DATA', 'EM_data_norm_' + model_type.lower() + '.pkl'))
CM_data = load_list(os.path.join(folderApplication, 'DATA', 'CM_data_norm_' + model_type.lower() + '.pkl'))

if sex != 'All':
    if sex == 'Females':
        indices_to_keep_EM = [i for i in range(len(covar_EM[0])) if all(array[i] != 'M' for array in covar_EM)]
        covar_EM = [[array[i] for i in indices_to_keep_EM] for array in covar_EM]
        age_EM, sex_EM, etiv_EM = covar_EM[0], covar_EM[1], covar_EM[2]

        indices_to_keep_CM = [i for i in range(len(covar_CM[0])) if all(array[i] != 'M' for array in covar_CM)]
        covar_CM = [[array[i] for i in indices_to_keep_CM] for array in covar_CM]
        age_CM, sex_CM, etiv_CM = covar_CM[0], covar_CM[1], covar_CM[2]

        indices_to_keep_HC = [i for i in range(len(covar_HC[0])) if all(array[i] != 'M' for array in covar_HC)]
        covar_HC = [[array[i] for i in indices_to_keep_HC] for array in covar_HC]
        age_HC, sex_HC, etiv_HC = covar_HC[0], covar_HC[1], covar_HC[2]

        EM_data = [arr[indices_to_keep_EM] for arr in EM_data]
        CM_data = [arr[indices_to_keep_CM] for arr in CM_data]
        HC_data = [arr[indices_to_keep_HC] for arr in HC_data]

        covars = ['etivs']

    if sex == 'Males':
        indices_to_keep_EM = [i for i in range(len(covar_EM[0])) if all(array[i] != 'F' for array in covar_EM)]
        covar_EM = [[array[i] for i in indices_to_keep_EM] for array in covar_EM]
        age_EM, sex_EM, etiv_EM = covar_EM[0], covar_EM[1], covar_EM[2]

        indices_to_keep_CM = [i for i in range(len(covar_CM[0])) if all(array[i] != 'F' for array in covar_CM)]
        covar_CM = [[array[i] for i in indices_to_keep_CM] for array in covar_CM]
        age_CM, sex_CM, etiv_CM = covar_CM[0], covar_CM[1], covar_CM[2]

        indices_to_keep_HC = [i for i in range(len(covar_HC[0])) if all(array[i] != 'F' for array in covar_HC)]
        covar_HC = [[array[i] for i in indices_to_keep_HC] for array in covar_HC]
        age_HC, sex_HC, etiv_HC = covar_HC[0], covar_HC[1], covar_HC[2]

        EM_data = [arr[indices_to_keep_EM] for arr in EM_data]
        CM_data = [arr[indices_to_keep_CM] for arr in CM_data]
        HC_data = [arr[indices_to_keep_HC] for arr in HC_data]

        covars = ['real_age', 'etivs']

else:
   # load data  saved
   age_EM, sex_EM, etiv_EM = covar_EM[0], covar_EM[1], covar_EM[2]
   age_CM, sex_CM, etiv_CM = covar_CM[0], covar_CM[1], covar_CM[2]
   age_HC, sex_HC, etiv_HC = covar_HC[0], covar_HC[1], covar_HC[2]

   covars = ['etivs', 'sex']

# for each folder, get the file names sorted
model_list = sorted([model for model in os.listdir(folder_models) if 'MLP_nfeats_'+str(n_features) in model])

# load the validation ages by fold, required for bias correction
age_val_folds = load_list(os.path.join(folderCreation, 'BiasCorrectionData', 'age_val_folds_'+model_type.lower()+'.pkl'))
pred_val_folds = load_list(os.path.join(folderCreation, 'BiasCorrectionData', 'pred_val_folds_'+model_type.lower()+'.pkl'))

print('[## INFO ##] Finished Loading.', end='\n\n')

pred_HC, pred_CM, pred_EM = [], [], []
print("[## INFO ##] Calculating results by split...")
pred_HC = [pickle.load(open(os.path.join(folder_models, model_list[i]), 'rb')).predict(HC_data[i]) for i in range(10)]
pred_CM = [pickle.load(open(os.path.join(folder_models, model_list[i]), 'rb')).predict(CM_data[i]) for i in range(10)]
pred_EM = [pickle.load(open(os.path.join(folder_models, model_list[i]), 'rb')).predict(EM_data[i]) for i in range(10)]
print("[## INFO ##] Calculated.", end='\n\n')

# Apply brain Age bias correction
pred_HC_corrected, pred_CM_corrected, pred_EM_corrected = \
    brain_age_bias_correction(age_val_folds, pred_val_folds, pred_HC, pred_CM, pred_EM)

# Average results Ensemble model
pred_avg_HC_cor = sum(pred_HC_corrected)/10
pred_avg_CM_cor = sum(pred_CM_corrected)/10
pred_avg_EM_cor = sum(pred_EM_corrected)/10

# calculate Brain Age Gap
BAG_HC_cor = pred_avg_HC_cor - age_HC
BAG_CM_cor = pred_avg_CM_cor - age_CM
BAG_EM_cor = pred_avg_EM_cor - age_EM

print("[INFO] MAE & Pearson r values for each group:")
mae_controls = mean_absolute_error(age_HC, pred_avg_HC_cor)
r_correlation_HC = stats.pearsonr(age_HC, pred_avg_HC_cor)[0]
print("MAE HC: {:.2f}".format(mae_controls) +"; r HC: {:.2f}".format(r_correlation_HC))
mae_EM = mean_absolute_error(age_EM, pred_avg_EM_cor)
r_correlation_EM = stats.pearsonr(age_EM, pred_avg_EM_cor)[0]
print("MAE EM: {:.2f}".format(mae_EM)+"; r EM: {:.2f}".format(r_correlation_EM))
mae_CM = mean_absolute_error(age_CM, pred_avg_CM_cor)
r_correlation_CM = stats.pearsonr(age_CM, pred_avg_CM_cor)[0]
print("MAE CM: {:.2f}".format(mae_CM)+"; r CM: {:.2f}".format(r_correlation_CM), end='\n\n')

# 1.- Check sex distribution differences across groups
print('[INFO] Sex - Xi-square test: ')
CM_sex_count = np.unique(sex_CM, return_index=False, return_inverse=False, return_counts=True)
EM_sex_count = np.unique(sex_EM, return_index=False, return_inverse=False, return_counts=True)
HC_sex_count = np.unique(sex_HC, return_index=False, return_inverse=False, return_counts=True)
obs = np.array([CM_sex_count[1], EM_sex_count[1], HC_sex_count[1]])
print("Xi2 statistic: {:.2f}".format(chi2_contingency(obs).statistic)+"; Xi2 p-value: {:.2f}".format(chi2_contingency(obs).pvalue), end='\n\n')

# 2.- Check normality of age across groups
print('[INFO] Check normality of age across groups (Kolmogorov-Smirnov test).')
mean, std = np.mean(age_EM), np.std(age_EM)
stat, pval = kstest(age_EM, 'norm', args=(mean, std))
print("KS-test statistic migr_ep: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval))
mean, std = np.mean(age_CM), np.std(age_CM)
stat, pval = kstest(age_CM, 'norm', args=(mean, std))
print("KS-test statistic migr_cr: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval))
mean, std = np.mean(age_HC), np.std(age_HC)
stat, pval = kstest(age_HC, 'norm', args=(mean, std))
print("KS-test statistic controls: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

# 3.- Check homocedasticity ages
print('[INFO] Age variance homocedasticity Levene\'s test')
stat, pval = levene(age_EM, age_CM, age_HC)
print("Levene statistic: {:.2f}".format(stat)+"; Levene p-value: {:.2f}".format(pval), end='\n\n')

# ANOVA
print('[INFO] ANOVA - age differences:')
stat, pval = f_oneway(age_EM, age_CM, age_HC)
print("ANOVA statistic: {:.2f}".format(stat)+"; ANOVA p-value: {:.2f}".format(pval), end='\n\n')


print("###############################################################")
print("[INFO] TEST Controls - Episodic Migraine - Chronic Migraine ###")
print("###############################################################", end='\n\n')
# 1.- Check Normality
print('[INFO] Check Brain Age Gap Normality (Kolmogorov-Smirnov test)')
mean, std = np.mean(BAG_HC_cor), np.std(BAG_HC_cor)
stat, pval = kstest(BAG_HC_cor, 'norm', args=(mean, std))
print("KS-test statistic HC: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval))
mean, std = np.mean(BAG_EM_cor), np.std(BAG_EM_cor)
stat, pval = kstest(BAG_EM_cor, 'norm', args=(mean, std))
print("KS-test statistic EM: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval))
mean, std = np.mean(BAG_CM_cor), np.std(BAG_CM_cor)
stat, pval = kstest(BAG_CM_cor, 'norm', args=(mean, std))
print("KS-test statistic CM: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

# 2.- Check homocedasticity
print('Check homocedasticity of variances across groups (Levene\'s Test):')
stat, pval = levene(BAG_HC_cor, BAG_EM_cor, BAG_CM_cor)
print("Levene statistic: {:.2f}".format(stat)+"; Levene p-value: {:.2f}".format(pval))
print('Variances:')
print("Variance Brain Age Gap HC: {:.2f}".format(np.var(BAG_HC_cor)))
print("Variance Brain Age Gap EM: {:.2f}".format(np.var(BAG_EM_cor)))
print("Variance Brain Age Gap CM: {:.2f}".format(np.var(BAG_CM_cor)), end='\n\n')

BAG = np.concatenate((BAG_HC_cor, BAG_EM_cor, BAG_CM_cor))
ages = np.concatenate((age_HC, age_EM, age_CM))
sexs = np.concatenate((sex_HC, sex_EM, sex_CM))
etivs = np.concatenate((etiv_HC, etiv_EM, etiv_CM))
sexos_cat = np.where(sexs == 'F', 1, 0)
type_bag = np.concatenate((np.repeat('HC', len(BAG_HC_cor)), np.repeat('EM', len(BAG_EM_cor)), np.repeat('CM', len(BAG_CM_cor))))

df_ancova = {'real_age': ages.tolist(), 'sex': sexos_cat, 'etivs': etivs.tolist(), 'BAG': BAG.tolist(), 'type': type_bag.tolist()}
df_ancova = pd.DataFrame.from_dict(df_ancova)

# statistical test
print("############################ ANCOVA RESULTS ############################")
print("########################################################################")
print(ancova(data=df_ancova, dv='BAG', covar=covars, between='type'), end='\n\n')

# Calculate Cohen's d for three groups pairwise
d_12 = pg.compute_effsize(x=BAG_HC_cor, y=BAG_EM_cor, eftype='cohen')
d_13 = pg.compute_effsize(x=BAG_HC_cor, y=BAG_CM_cor, eftype='cohen')
d_23 = pg.compute_effsize(x=BAG_EM_cor, y=BAG_CM_cor, eftype='cohen')

print(f"Cohen's d between HC and EM: {d_12}")
print(f"Cohen's d between HC and CM: {d_13}")
print(f"Cohen's d between EM and CM: {d_23}", end='\n\n')

print("############################################")
print("[INFO] TEST Controls - Episodic Migraine ###")
print("############################################", end='\n\n')
# 1.- Check Normality
print('[INFO] Check Brain Age Gap Normality (Kolmogorov-Smirnov test)')
mean, std = np.mean(BAG_HC_cor), np.std(BAG_HC_cor)
stat, pval = kstest(BAG_HC_cor, 'norm', args=(mean, std))
print("KS-test statistic HC: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval))
mean, std = np.mean(BAG_EM_cor), np.std(BAG_EM_cor)
stat, pval = kstest(BAG_EM_cor, 'norm', args=(mean, std))
print("KS-test statistic EM: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

# 2.- Check homocedasticity
print('Check homocedasticity of variances across groups (Levene\'s Test):')
stat, pval = levene(BAG_EM_cor, BAG_HC_cor)
print("Levene statistic: {:.2f}".format(stat)+"; Levene p-value: {:.2f}".format(pval))
print('Variances:')
print("Variance Brain Age Gap EM: {:.2f}".format(np.var(BAG_EM_cor)))
print("Variance Brain Age Gap HC: {:.2f}".format(np.var(BAG_HC_cor)), end='\n\n')

BAG = np.concatenate((BAG_HC_cor, BAG_EM_cor))
ages = np.concatenate((age_HC, age_EM))
sexs = np.concatenate((sex_HC, sex_EM))
etivs = np.concatenate((etiv_HC, etiv_EM))
sexos_cat = np.where(sexs == 'F', 1, 0)
type_bag = np.concatenate((np.repeat('HC', len(BAG_HC_cor)), np.repeat('EM', len(BAG_EM_cor))))

df_ancova = {'real_age': ages.tolist(), 'sex': sexos_cat, 'etivs': etivs.tolist(), 'BAG': BAG.tolist(), 'type': type_bag.tolist()}
df_ancova = pd.DataFrame.from_dict(df_ancova)

# statistical test
print("############################ ANCOVA RESULTS ############################")
print("########################################################################")
print(ancova(data=df_ancova, dv='BAG', covar=covars, between='type'), end='\n\n')

print("###########################################")
print("[INFO] TEST Controls - Chronic Migraine ###")
print("###########################################", end='\n\n')

# 1.- check normality
print('[INFO] Check Brain Age Gap Normality (Kolmogorov-Smirnov test):')
mean, std = np.mean(BAG_EM_cor), np.std(BAG_EM_cor)
stat, pval = kstest(BAG_EM_cor, 'norm', args=(mean, std))
print("KS-test statistic migr_cr: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

# 2.- check homocedasticity
print('Check homocedasticity of variances across groups (Levene\'s Test):')
stat, pval = levene(BAG_CM_cor, BAG_HC_cor)
print("Levene statistic: {:.2f}".format(stat)+"; Levene p-value: {:.2f}".format(pval))
print('Variances:')
print("Variance Brain Age Gap CM: {:.2f}".format(np.var(BAG_CM_cor)))
print("Variance Brain Age Gap HC: {:.2f}".format(np.var(BAG_HC_cor)), end='\n\n')

BAG = np.concatenate((BAG_HC_cor, BAG_CM_cor))
ages = np.concatenate((age_HC, age_CM))
sexs = np.concatenate((sex_HC, sex_CM))
etivs = np.concatenate((etiv_HC, etiv_CM))
sexos_cat = np.where(sexs == 'F', 1, 0)
type_bag = np.concatenate((np.repeat('HC', len(BAG_HC_cor)), np.repeat('CM', len(BAG_CM_cor))))

df_ancova = {'real_age': ages.tolist(), 'sex': sexos_cat, 'etivs': etivs.tolist(), 'BAG': BAG.tolist(), 'type': type_bag.tolist()}
df_ancova = pd.DataFrame.from_dict(df_ancova)

# 3.- statistical test
print("############################ ANCOVA RESULTS ############################")
print("########################################################################")
print(ancova(data=df_ancova, dv='BAG', covar=covars, between='type'), end='\n\n')

print("###############################################")
print("[INFO] Episodic Migraine - Chronic Migraine ###")
print("###############################################", end='\n\n')
# 1.- check normality
print('[INFO] Check Brain Age Gap Normality (Kolmogorov-Smirnov test):')
print('Already Tested!', end='\n\n')

# 2.- check homocedasticity
print('Check homocedasticity of variances across groups (Levene\'s Test):')
stat, pval = levene(BAG_EM_cor, BAG_CM_cor)
print("Levene statistic: {:.2f}".format(stat)+"; Levene p-value: {:.2f}".format(pval))
print('Variances:')
print("Variance Brain Age Gap migr ep: {:.2f}".format(np.var(BAG_EM_cor)))
print("Variance Brain Age Gap migr cr: {:.2f}".format(np.var(BAG_CM_cor)), end='\n\n')

BAG = np.concatenate((BAG_EM_cor, BAG_CM_cor))
ages = np.concatenate((age_EM, age_CM))
sexs = np.concatenate((sex_EM, sex_CM))
etivs = np.concatenate((etiv_EM, etiv_CM))
sexos_cat = np.where(sexs == 'F', 1, 0)
type_bag = np.concatenate((np.repeat('EM', len(BAG_EM_cor)), np.repeat('CM', len(BAG_CM_cor))))

df_ancova = {'real_age': ages.tolist(), 'sex': sexos_cat, 'etivs': etivs.tolist(), 'BAG': BAG.tolist(), 'type': type_bag.tolist()}
df_ancova = pd.DataFrame.from_dict(df_ancova)

# 3.- statistical test
print("############################ ANCOVA RESULTS ############################")
print("########################################################################")
print(ancova(data=df_ancova, dv='BAG', covar=covars, between='type'), end ='\n\n')

################### Statistical differences in clinical variables ###################
print('[INFO] Statistical differences in clinical variables:')

clin_CM = load_list(os.path.join(folderApplication, 'Demographics_and_Clinical', 'clin_CM_'+model_type.lower()+'.pkl'))
clin_EM = load_list(os.path.join(folderApplication, 'Demographics_and_Clinical', 'clin_EM_'+model_type.lower()+'.pkl'))

# Migraine Duration
print('Migraine duration in episodic migraine (mean & std):')
print('Mean: {:.2f}'.format(np.mean(clin_EM[2])))
print('Std: {:.2f}'.format(np.mean(np.std(clin_EM[2]))), end='\n\n')

print('Migraine duration in chronic migraine (mean & std):')
print('Mean: {:.2f}'.format(np.mean(clin_CM[2])))
print('Std: {:.2f}'.format(np.mean(np.std(clin_CM[2]))), end='\n\n')

print('[INFO] Check migraine duration diffrences between groups:')
# Conditions
# 1.- Check for normality
print('Check migraine duration normality in episodic migraine group (Kolmogorov-Smirnov test):')
mean, std = np.mean(clin_EM[2]), np.std(clin_EM[2])
stat, pval = kstest(clin_EM[2], 'norm', args=(mean, std))
print("KS-test statistic: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

print('[INFO] Check migraine duration normality in chronic migraine group (Kolmogorov-Smirnov test):')
mean, std = np.mean(clin_CM[2]), np.std(clin_CM[2])
stat, pval = kstest(clin_CM[2], 'norm', args=(mean, std))
print("KS-test statistic: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

# 2.- Check for homocedasticity of variances
print('Check homocedasticity of variances across groups (Levene\'s Test):')
stat, pval = levene(clin_EM[2], clin_CM[2])
print("Levene statistic: {:.2f}".format(stat)+"; Levene p-value: {:.2f}".format(pval))
print('Variances:')
print("Variance miraine duration migr ep: {:.2f}".format(np.var(clin_EM[2])))
print("Variance miraine duration migr cr: {:.2f}".format(np.var(clin_CM[2])), end='\n\n')

# 3.- Perform t-test
print('Check T-test for differences:')
stat, pval = stats.ttest_ind(a=clin_EM[2], b=clin_CM[2], equal_var=True)
print("T-test statistic: {:.2f}".format(stat)+"; t-test p-value: {:.2f}".format(pval))

# Headache Frequency
print('Headache Frequency in episodic migraine (mean & std):')
print('Mean: {:.2f}'.format(np.mean(clin_EM[3])))
print('Std: {:.2f} '.format(np.mean(np.std(clin_EM[3]))), end='\n\n')

print('Headache Frequency in chronic migraine (mean & std):')
print('Mean: {:.2f}'.format(np.mean(clin_CM[3])))
print('Std: {:.2f}'.format(np.mean(np.std(clin_CM[3]))), end='\n\n')

print('[INFO] Check migraine duration diffrences between groups:')
# Conditions
# 1.- Check for normality
print('Check Headache Frequency normality in episodic migraine group (Kolmogorov-Smirnov test):')
mean, std = np.mean(clin_EM[3]), np.std(clin_EM[3])
stat, pval = kstest(clin_EM[3], 'norm', args=(mean, std))
print("KS-test statistic: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

print('[INFO] Check Headache Frequency normality in chronic migraine group (Kolmogorov-Smirnov test):')
mean, std = np.mean(clin_CM[3]), np.std(clin_CM[3])
stat, pval = kstest(clin_CM[3], 'norm', args=(mean, std))
print("KS-test statistic: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

# 2.- Check for homocedasticity of variances
print('Check homocedasticity of variances across groups (Levene\'s Test):')
stat, pval = levene(clin_EM[3], clin_CM[3])
print("Levene statistic: {:.2f}".format(stat)+"; Levene p-value: {:.2f}".format(pval))
print('Variances:')
print("Variance Headache Frequency migr ep: {:.2f}".format(np.var(clin_EM[3])))
print("Variance Headache Frequency migr cr: {:.2f}".format(np.var(clin_CM[3])), end='\n\n')

# 3.- U-Mann Whitney
stat, pval = stats.mannwhitneyu(clin_EM[3], clin_CM[3], method="auto")
print("U-Mann Whitney statistic: {:.2f}".format(stat)+"; U-Mann Whitney p-value: {:.2f}".format(pval), end='\n\n')

# Migraine Frequency
print('Migraine Frequency in episodic migraine (mean & std):')
print('Mean: {:.2f}'.format(np.mean(clin_EM[4])))
print('Std: {:.2f}'.format(np.mean(np.std(clin_EM[4]))), end='\n\n')

print('Migraine Frequency in chronic migraine (mean & std):')
print('Mean: {:.2f}'.format(np.mean(clin_CM[4])))
print('Std: {:.2f}'.format(np.mean(np.std(clin_CM[4]))), end='\n\n')

print('[INFO] Check migraine duration diffrences between groups:')
# Conditions
# 1.- Check for normality
print('Check Migraine Frequency normality in episodic migraine group (Kolmogorov-Smirnov test):')
mean, std = np.mean(clin_EM[4]), np.std(clin_EM[4])
stat, pval = kstest(clin_EM[4], 'norm', args=(mean, std))
print("KS-test statistic: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

print('[INFO] Check Migraine Frequency normality in chronic migraine group (Kolmogorov-Smirnov test):')
mean, std = np.mean(clin_CM[4]), np.std(clin_CM[4])
stat, pval = kstest(clin_CM[4], 'norm', args=(mean, std))
print("KS-test statistic: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

# 2.- Check for homocedasticity of variances
print('Check homocedasticity of variances across groups (Levene\'s Test):')
stat, pval = levene(clin_EM[4], clin_CM[4])
print("Levene statistic: {:.2f}".format(stat)+"; Levene p-value: {:.2f}".format(pval))
print('Variances:')
print("Variance Migraine Frequency migr ep: {:.2f}".format(np.var(clin_EM[4])))
print("Variance Migraine Frequency migr cr: {:.2f}".format(np.var(clin_CM[4])), end='\n\n')

# 3.- U-Mann Whitney
stat, pval = stats.mannwhitneyu(clin_EM[4], clin_CM[4], method="auto")
print("U-Mann Whitney statistic: {:.2f}".format(stat)+"; U-Mann Whitney p-value: {:.2f}".format(pval), end='\n\n')

################### OTHER PLOTS ###################
print('#################################################################################')
print('##################################### PLOTS #####################################')
print('#################################################################################', end='\n\n')

# Prepare the data scatter plot
BAG = np.concatenate((BAG_EM_cor, BAG_CM_cor, BAG_HC_cor))
real_ages = np.concatenate((age_EM, age_CM, age_HC))
predicted_ages = np.concatenate((pred_avg_EM_cor, pred_avg_CM_cor, pred_avg_HC_cor))
type_bag = np.concatenate((np.repeat('migr_ep', len(BAG_EM_cor)), np.repeat('migr_cr', len(BAG_CM_cor)),  np.repeat('controls', len(BAG_HC_cor))))
df_scatter = {'real_ages': real_ages.tolist(), 'predicted_ages': predicted_ages.tolist(), 'type': type_bag.tolist()}
df_scatter = pd.DataFrame.from_dict(df_scatter)

x=df_scatter[df_scatter['type'] =='controls']['real_ages'].values
y=df_scatter[df_scatter['type'] =='controls']['predicted_ages'].values
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#789837', alpha=1, marker="^", label="Controls")
plt.plot(x, poly1d_fn(x), color='#789837')

x=df_scatter[df_scatter['type'] =='migr_ep']['real_ages'].values
y=df_scatter[df_scatter['type'] =='migr_ep']['predicted_ages'].values
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#5481A6', alpha=1, marker="*", label="Episodic Migraine")
plt.plot(x, poly1d_fn(x), color='#5481A6')

x=df_scatter[df_scatter['type'] =='migr_cr']['real_ages'].values
y=df_scatter[df_scatter['type'] =='migr_cr']['predicted_ages'].values
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#FEA32A', alpha=1, marker="d", label="Chronic Migraine")
plt.plot(x, poly1d_fn(x), color='#FEA32A')

X_plot = np.linspace(18, 60, 42)
Y_plot = np.linspace(18, 60, 42)
plt.legend(loc="upper left")
plt.plot(X_plot, Y_plot, '--k', linewidth=0.7)
plt.title('Brain Age', fontweight='bold')
plt.ylabel('Predicted Age', fontweight='bold')
plt.xlabel('Real Age', fontweight='bold')
plt.show()

# Prepare data violin plot
# Hago el violin plot
BAG = np.concatenate((BAG_EM_cor, BAG_CM_cor, BAG_HC_cor))
type_bag = np.concatenate((np.repeat('EM', len(BAG_EM_cor)), np.repeat('CM', len(BAG_CM_cor)),  np.repeat('HC', len(BAG_HC_cor))))

d = {'BAG': BAG.tolist(), 'type': type_bag.tolist()}
dataframe_violin_plot = pd.DataFrame(data=d)
sns.violinplot(data=dataframe_violin_plot, x="type", y="BAG")
plt.title('Brain Age Gap distribution', fontweight='bold')
plt.ylabel('Brain Age Gap (years)', fontweight='bold')
plt.xlabel('Group', fontweight='bold')
plt.ylim(-35, 35)
plt.show()

# prepare data scatter plot brain age gap clinical variables
# Migraine duration
x = clin_CM[2]
y = clin_CM[1]
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#FF5555', alpha=1, marker="o")
plt.plot(x, poly1d_fn(x), color='#FF5555', linewidth=2)

x = clin_EM[2]
y = clin_EM[1]
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#00CCFF', alpha=1, marker="o")
plt.plot(x, poly1d_fn(x), color='#00CCFF', linewidth=2)

x = np.concatenate((clin_EM[2],  clin_CM[2]))
y = np.concatenate((clin_EM[1],  clin_CM[1]))
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.plot(x, poly1d_fn(x), color='#FFDD55', linewidth=2)

plt.xlabel('Migraine duration (years)', fontweight='bold')
plt.ylabel('BrainAGE (years)', fontweight='bold')
plt.show()


# Headache Frequency
x = clin_CM[3]
y = clin_CM[1]
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#FF5555', alpha=1, marker="o")
plt.plot(x, poly1d_fn(x), color='#FF5555', linewidth=2)

x = clin_EM[3]
y = clin_EM[1]
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#00CCFF', alpha=1, marker="o")
plt.plot(x, poly1d_fn(x), color='#00CCFF', linewidth=2)

x = np.concatenate((clin_EM[3],  clin_CM[3]))
y = np.concatenate((clin_EM[1],  clin_CM[1]))
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.plot(x, poly1d_fn(x), color='#FFDD55', linewidth=2)

plt.xlabel('Headache frequency (bouts/month)', fontweight='bold')
plt.ylabel('BrainAGE (years)', fontweight='bold')
plt.show()

# Migraine Frequency
x = clin_CM[4]
y = clin_CM[1]
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#FF5555', alpha=1, marker="o")
plt.plot(x, poly1d_fn(x), color='#FF5555', linewidth=2)

x = clin_EM[4]
y = clin_EM[1]
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#00CCFF', alpha=1, marker="o")
plt.plot(x, poly1d_fn(x), color='#00CCFF', linewidth=2)

x = np.concatenate((clin_EM[4],  clin_CM[4]))
y = np.concatenate((clin_EM[1],  clin_CM[1]))
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.plot(x, poly1d_fn(x), color='#FFDD55', linewidth=2)

plt.xlabel('Migraine frequency (bouts/month)', fontweight='bold')
plt.ylabel('BrainAGE (years)', fontweight='bold')
plt.show()

# Chronic migraine duration
x = clin_CM[5]
y = clin_CM[1]
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#FF5555', alpha=1, marker="o")
plt.plot(x, poly1d_fn(x), color='#FF5555', linewidth=2)
plt.xlabel('Chronic migraine duration (months)', fontweight='bold')
plt.ylabel('BrainAGE (years)', fontweight='bold')
plt.show()

print("#################### END SCRIPT ####################")
print("####################################################")
