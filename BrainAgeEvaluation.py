from Utils import *

import os
import configparser

from sklearn.metrics import mean_absolute_error

from pingouin import ancova

from scipy.stats import f_oneway
from scipy.stats import kstest, levene, chi2_contingency
from scipy import stats

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

# load data  saved
covar_migr_ep = load_list(os.path.join(folderApplication, 'Demographics_and_Clinical', 'covariates_migr_ep.pkl'))
edad_migr_ep = covar_migr_ep[0]
sex_migr_ep = covar_migr_ep[1]
etiv_migr_ep = covar_migr_ep[2]

covar_migr_cr = load_list(os.path.join(folderApplication, 'Demographics_and_Clinical', 'covariates_migr_cr.pkl'))
edad_migr_cr = covar_migr_cr[0]
sex_migr_cr = covar_migr_cr[1]
etiv_migr_cr = covar_migr_cr[2]

covar_controls = load_list(os.path.join(folderApplication, 'Demographics_and_Clinical', 'covariates_controls.pkl'))
edad_controls_sel = covar_controls[0]
sex_control_sel = covar_controls[1]
etiv_control_sel = covar_controls[2]

# for each folder, get the file names sorted
model_list = sorted([model for model in os.listdir(folder_models) if 'MLP_nfeats_'+str(n_features) in model])

# load the validation ages by fold, required for bias correction
age_val_folds = load_list(os.path.join(folderCreation, 'BiasCorrectionData', 'age_val_folds_'+model_type.lower()+'.pkl'))
pred_val_folds = load_list(os.path.join(folderCreation, 'BiasCorrectionData', 'pred_val_folds_'+model_type.lower()+'.pkl'))

# load normalized controls, episodic migraine data and crhonic migraine data
control_data = load_list(os.path.join(folderApplication, 'DATA', 'control_data_norm_'+model_type.lower()+'.pkl'))
migr_ep_data = load_list(os.path.join(folderApplication, 'DATA', 'migr_ep_data_norm_'+model_type.lower()+'.pkl'))
migr_cr_data = load_list(os.path.join(folderApplication, 'DATA', 'migr_cr_data_norm_'+model_type.lower()+'.pkl'))

print('[## INFO ##] Finished Loading.', end='\n\n')

pred_controls, pred_migr_cr, pred_migr_ep = [], [], []
print("[## INFO ##] Calculating results by split...")
pred_controls = [pickle.load(open(os.path.join(folder_models, model_list[i]), 'rb')).predict(control_data[i]) for i in range(10)]
pred_migr_cr = [pickle.load(open(os.path.join(folder_models, model_list[i]), 'rb')).predict(migr_cr_data[i]) for i in range(10)]
pred_migr_ep = [pickle.load(open(os.path.join(folder_models, model_list[i]), 'rb')).predict(migr_ep_data[i]) for i in range(10)]
print("[## INFO ##] Calculated.", end='\n\n')

# Apply brain Age bias correction
pred_controls_corrected, pred_migr_cr_corrected, pred_migr_ep_corrected = \
    brain_age_bias_correction(age_val_folds, pred_val_folds, pred_controls, pred_migr_cr, pred_migr_ep)

# Average results Ensemble model
pred_avg_controls_cor = sum(pred_controls_corrected)/10
pred_avg_migr_cr_cor = sum(pred_migr_cr_corrected)/10
pred_avg_migr_ep_cor = sum(pred_migr_ep_corrected)/10

# calculate Brain Age Gap
BAG_controls_cor = pred_avg_controls_cor - edad_controls_sel
BAG_migr_cr_cor = pred_avg_migr_cr_cor - edad_migr_cr
BAG_migr_ep_cor = pred_avg_migr_ep_cor - edad_migr_ep

print("[INFO] MAE & Pearson r values for each group:")
mae_controls = mean_absolute_error(edad_controls_sel, pred_avg_controls_cor)
r_correlation_controls = stats.pearsonr(edad_controls_sel, pred_avg_controls_cor)[0]
print("MAE Controls: {:.2f}".format(mae_controls) +"; r Controls: {:.2f}".format(r_correlation_controls))
mae_migr_ep = mean_absolute_error(edad_migr_ep, pred_avg_migr_ep_cor)
r_correlation_migr_ep = stats.pearsonr(edad_migr_ep, pred_avg_migr_ep_cor)[0]
print("MAE migr ep: {:.2f}".format(mae_migr_ep)+"; r migr ep: {:.2f}".format(r_correlation_migr_ep))
mae_migr_cr = mean_absolute_error(edad_migr_cr, pred_avg_migr_cr_cor)
r_correlation_migr_cr = stats.pearsonr(edad_migr_cr, pred_avg_migr_cr_cor)[0]
print("MAE migr cr: {:.2f}".format(mae_migr_cr)+"; r migr cr: {:.2f}".format(r_correlation_migr_cr), end='\n\n')

# 1.- Check sex distribution differences across groups
print('[INFO] Sex - Xi-square test: ')
migr_cron_sex_count = np.unique(sex_migr_cr, return_index=False, return_inverse=False, return_counts=True)
migr_epi_sex_count = np.unique(sex_migr_ep, return_index=False, return_inverse=False, return_counts=True)
migr_control_sex_count = np.unique(sex_control_sel, return_index=False, return_inverse=False, return_counts=True)
obs = np.array([migr_cron_sex_count[1], migr_epi_sex_count[1], migr_control_sex_count[1]])
print("Xi2 statistic: {:.2f}".format(chi2_contingency(obs).statistic)+"; Xi2 p-value: {:.2f}".format(chi2_contingency(obs).pvalue), end='\n\n')

# 2.- Check normality of age across groups
print('[INFO] Check normality of age across groups (Kolmogorov-Smirnov test).')
mean, std = np.mean(edad_migr_ep), np.std(edad_migr_ep)
stat, pval = kstest(edad_migr_ep, 'norm', args=(mean, std))
print("KS-test statistic migr_ep: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval))
mean, std = np.mean(edad_migr_cr), np.std(edad_migr_cr)
stat, pval = kstest(edad_migr_cr, 'norm', args=(mean, std))
print("KS-test statistic migr_cr: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval))
mean, std = np.mean(edad_controls_sel), np.std(edad_controls_sel)
stat, pval = kstest(edad_controls_sel, 'norm', args=(mean, std))
print("KS-test statistic controls: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

# 3.- Check homocedasticity ages
print('[INFO] Age variance homocedasticity Levene\'s test')
stat, pval = levene(edad_migr_ep, edad_migr_cr, edad_controls_sel)
print("Levene statistic: {:.2f}".format(stat)+"; Levene p-value: {:.2f}".format(pval), end='\n\n')

# ANOVA
print('[INFO] ANOVA - age differences:')
stat, pval = f_oneway(edad_migr_ep, edad_migr_cr, edad_controls_sel)
print("ANOVA statistic: {:.2f}".format(stat)+"; ANOVA p-value: {:.2f}".format(pval), end='\n\n')

print("############################################")
print("[INFO] TEST Controls - Episodic Migraine ###")
print("############################################", end='\n\n')
# 1.- Check Normality
print('[INFO] Check Brain Age Gap Normality (Kolmogorov-Smirnov test)')
mean, std = np.mean(BAG_controls_cor), np.std(BAG_controls_cor)
stat, pval = kstest(BAG_controls_cor, 'norm', args=(mean, std))
print("KS-test statistic controls: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval))
mean, std = np.mean(BAG_migr_ep_cor), np.std(BAG_migr_ep_cor)
stat, pval = kstest(BAG_migr_ep_cor, 'norm', args=(mean, std))
print("KS-test statistic migr_ep: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

# 2.- Check homocedasticity
print('Check homocedasticity of variances across groups (Levene\'s Test):')
stat, pval = levene(BAG_migr_ep_cor, BAG_controls_cor)
print("Levene statistic: {:.2f}".format(stat)+"; Levene p-value: {:.2f}".format(pval))
print('Variances:')
print("Variance Brain Age Gap migr ep: {:.2f}".format(np.var(BAG_migr_ep_cor)))
print("Variance Brain Age Gap controls: {:.2f}".format(np.var(BAG_controls_cor)), end='\n\n')

BAG = np.concatenate((BAG_controls_cor, BAG_migr_ep_cor))
ages = np.concatenate((edad_controls_sel, edad_migr_ep))
sexs = np.concatenate((sex_control_sel, sex_migr_ep))
etivs = np.concatenate((etiv_control_sel, etiv_migr_ep))
sexos_cat = np.where(sexs == 'F', 1, 0)
type_bag = np.concatenate((np.repeat('controls', len(BAG_controls_cor)), np.repeat('migr_ep', len(BAG_migr_ep_cor))))

df_ancova = {'edad_real': ages.tolist(), 'sexos': sexos_cat, 'etivs': etivs.tolist(), 'BAG': BAG.tolist(), 'type': type_bag.tolist()}
df_ancova = pd.DataFrame.from_dict(df_ancova)

# statistical test
print("############################ ANCOVA RESULTS ############################")
print("########################################################################")
print(ancova(data=df_ancova, dv='BAG', covar=['etivs', 'sexos'], between='type'), end='\n\n')

print("###########################################")
print("[INFO] TEST Controls - Chronic Migraine ###")
print("###########################################", end='\n\n')

# 1.- check normality
print('[INFO] Check Brain Age Gap Normality (Kolmogorov-Smirnov test):')
mean, std = np.mean(BAG_migr_ep_cor), np.std(BAG_migr_ep_cor)
stat, pval = kstest(BAG_migr_ep_cor, 'norm', args=(mean, std))
print("KS-test statistic migr_cr: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

# 2.- check homocedasticity
print('Check homocedasticity of variances across groups (Levene\'s Test):')
stat, pval = levene(BAG_migr_cr_cor, BAG_controls_cor)
print("Levene statistic: {:.2f}".format(stat)+"; Levene p-value: {:.2f}".format(pval))
print('Variances:')
print("Variance Brain Age Gap migr cr: {:.2f}".format(np.var(BAG_migr_cr_cor)))
print("Variance Brain Age Gap controls: {:.2f}".format(np.var(BAG_controls_cor)), end='\n\n')

BAG = np.concatenate((BAG_controls_cor, BAG_migr_cr_cor))
ages = np.concatenate((edad_controls_sel, edad_migr_cr))
sexs = np.concatenate((sex_control_sel, sex_migr_cr))
etivs = np.concatenate((etiv_control_sel, etiv_migr_cr))
sexos_cat = np.where(sexs == 'F', 1, 0)
type_bag = np.concatenate((np.repeat('controls', len(BAG_controls_cor)), np.repeat('migr_cr', len(BAG_migr_cr_cor))))

df_ancova = {'edad_real': ages.tolist(), 'sexos': sexos_cat, 'etivs': etivs.tolist(), 'BAG': BAG.tolist(), 'type': type_bag.tolist()}
df_ancova = pd.DataFrame.from_dict(df_ancova)

# 3.- statistical test
print("############################ ANCOVA RESULTS ############################")
print("########################################################################")
print(ancova(data=df_ancova, dv='BAG', covar=['etivs', 'sexos'], between='type'), end='\n\n')

print("###############################################")
print("[INFO] Episodic Migraine - Chronic Migraine ###")
print("###############################################", end='\n\n')
# 1.- check normality
print('[INFO] Check Brain Age Gap Normality (Kolmogorov-Smirnov test):')
print('Already Tested!', end='\n\n')

# 2.- check homocedasticity
print('Check homocedasticity of variances across groups (Levene\'s Test):')
stat, pval = levene(BAG_migr_ep_cor, BAG_migr_cr_cor)
print("Levene statistic: {:.2f}".format(stat)+"; Levene p-value: {:.2f}".format(pval))
print('Variances:')
print("Variance Brain Age Gap migr ep: {:.2f}".format(np.var(BAG_migr_ep_cor)))
print("Variance Brain Age Gap migr cr: {:.2f}".format(np.var(BAG_migr_cr_cor)), end='\n\n')

BAG = np.concatenate((BAG_migr_ep_cor, BAG_migr_cr_cor))
ages = np.concatenate((edad_migr_ep, edad_migr_cr))
sexs = np.concatenate((sex_migr_ep, sex_migr_cr))
etivs = np.concatenate((etiv_migr_ep, etiv_migr_cr))
sexos_cat = np.where(sexs == 'F', 1, 0)
type_bag = np.concatenate((np.repeat('migr_ep', len(BAG_migr_ep_cor)), np.repeat('migr_cr', len(BAG_migr_cr_cor))))

df_ancova = {'edad_real': ages.tolist(), 'sexos': sexos_cat, 'etivs': etivs.tolist(), 'BAG': BAG.tolist(), 'type': type_bag.tolist()}
df_ancova = pd.DataFrame.from_dict(df_ancova)

# 3.- statistical test
print("############################ ANCOVA RESULTS ############################")
print("########################################################################")
print(ancova(data=df_ancova, dv='BAG', covar=['etivs', 'sexos'], between='type'), end ='\n\n')

################### Statistical differences in clinical variables ###################
print('[INFO] Statistical differences in clinical variables:')

clin_migr_cr = load_list(os.path.join(folderApplication, 'Demographics_and_Clinical', 'clin_migr_cr_'+model_type.lower()+'.pkl'))
clin_migr_ep = load_list(os.path.join(folderApplication, 'Demographics_and_Clinical', 'clin_migr_ep_'+model_type.lower()+'.pkl'))

# Migraine Duration
print('Migraine duration in episodic migraine (mean & std):')
print('Mean: {:.2f}'.format(np.mean(clin_migr_ep[2])))
print('Std: {:.2f}'.format(np.mean(np.std(clin_migr_ep[2]))), end='\n\n')

print('Migraine duration in chronic migraine (mean & std):')
print('Mean: {:.2f}'.format(np.mean(clin_migr_cr[2])))
print('Std: {:.2f}'.format(np.mean(np.std(clin_migr_cr[2]))), end='\n\n')

print('[INFO] Check migraine duration diffrences between groups:')
# Conditions
# 1.- Check for normality
print('Check migraine duration normality in episodic migraine group (Kolmogorov-Smirnov test):')
mean, std = np.mean(clin_migr_ep[2]), np.std(clin_migr_ep[2])
stat, pval = kstest(clin_migr_ep[2], 'norm', args=(mean, std))
print("KS-test statistic: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

print('[INFO] Check migraine duration normality in chronic migraine group (Kolmogorov-Smirnov test):')
mean, std = np.mean(clin_migr_cr[2]), np.std(clin_migr_cr[2])
stat, pval = kstest(clin_migr_cr[2], 'norm', args=(mean, std))
print("KS-test statistic: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

# 2.- Check for homocedasticity of variances
print('Check homocedasticity of variances across groups (Levene\'s Test):')
stat, pval = levene(clin_migr_ep[2], clin_migr_cr[2])
print("Levene statistic: {:.2f}".format(stat)+"; Levene p-value: {:.2f}".format(pval))
print('Variances:')
print("Variance miraine duration migr ep: {:.2f}".format(np.var(clin_migr_ep[2])))
print("Variance miraine duration migr cr: {:.2f}".format(np.var(clin_migr_cr[2])), end='\n\n')

# 3.- Perform t-test
print('Check T-test for differences:')
stat, pval = stats.ttest_ind(a=clin_migr_ep[2], b=clin_migr_cr[2], equal_var=True)
print("T-test statistic: {:.2f}".format(stat)+"; t-test p-value: {:.2f}".format(pval))

# Headache Frequency
print('Headache Frequency in episodic migraine (mean & std):')
print('Mean: {:.2f}'.format(np.mean(clin_migr_ep[3])))
print('Std: {:.2f} '.format(np.mean(np.std(clin_migr_ep[3]))), end='\n\n')

print('Headache Frequency in chronic migraine (mean & std):')
print('Mean: {:.2f}'.format(np.mean(clin_migr_cr[3])))
print('Std: {:.2f}'.format(np.mean(np.std(clin_migr_cr[3]))), end='\n\n')

print('[INFO] Check migraine duration diffrences between groups:')
# Conditions
# 1.- Check for normality
print('Check Headache Frequency normality in episodic migraine group (Kolmogorov-Smirnov test):')
mean, std = np.mean(clin_migr_ep[3]), np.std(clin_migr_ep[3])
stat, pval = kstest(clin_migr_ep[3], 'norm', args=(mean, std))
print("KS-test statistic: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

print('[INFO] Check Headache Frequency normality in chronic migraine group (Kolmogorov-Smirnov test):')
mean, std = np.mean(clin_migr_cr[3]), np.std(clin_migr_cr[3])
stat, pval = kstest(clin_migr_cr[3], 'norm', args=(mean, std))
print("KS-test statistic: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

# 2.- Check for homocedasticity of variances
print('Check homocedasticity of variances across groups (Levene\'s Test):')
stat, pval = levene(clin_migr_ep[3], clin_migr_cr[3])
print("Levene statistic: {:.2f}".format(stat)+"; Levene p-value: {:.2f}".format(pval))
print('Variances:')
print("Variance Headache Frequency migr ep: {:.2f}".format(np.var(clin_migr_ep[3])))
print("Variance Headache Frequency migr cr: {:.2f}".format(np.var(clin_migr_cr[3])), end='\n\n')

# 3.- U-Mann Whitney
stat, pval = stats.mannwhitneyu(clin_migr_ep[3], clin_migr_cr[3], method="auto")
print("U-Mann Whitney statistic: {:.2f}".format(stat)+"; U-Mann Whitney p-value: {:.2f}".format(pval), end='\n\n')

# Migraine Frequency
print('Migraine Frequency in episodic migraine (mean & std):')
print('Mean: {:.2f}'.format(np.mean(clin_migr_ep[4])))
print('Std: {:.2f}'.format(np.mean(np.std(clin_migr_ep[4]))), end='\n\n')

print('Migraine Frequency in chronic migraine (mean & std):')
print('Mean: {:.2f}'.format(np.mean(clin_migr_cr[4])))
print('Std: {:.2f}'.format(np.mean(np.std(clin_migr_cr[4]))), end='\n\n')

print('[INFO] Check migraine duration diffrences between groups:')
# Conditions
# 1.- Check for normality
print('Check Migraine Frequency normality in episodic migraine group (Kolmogorov-Smirnov test):')
mean, std = np.mean(clin_migr_ep[4]), np.std(clin_migr_ep[4])
stat, pval = kstest(clin_migr_ep[4], 'norm', args=(mean, std))
print("KS-test statistic: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

print('[INFO] Check Migraine Frequency normality in chronic migraine group (Kolmogorov-Smirnov test):')
mean, std = np.mean(clin_migr_cr[4]), np.std(clin_migr_cr[4])
stat, pval = kstest(clin_migr_cr[4], 'norm', args=(mean, std))
print("KS-test statistic: {:.2f}".format(stat)+"; KS-test p-value: {:.2f}".format(pval), end='\n\n')

# 2.- Check for homocedasticity of variances
print('Check homocedasticity of variances across groups (Levene\'s Test):')
stat, pval = levene(clin_migr_ep[4], clin_migr_cr[4])
print("Levene statistic: {:.2f}".format(stat)+"; Levene p-value: {:.2f}".format(pval))
print('Variances:')
print("Variance Migraine Frequency migr ep: {:.2f}".format(np.var(clin_migr_ep[4])))
print("Variance Migraine Frequency migr cr: {:.2f}".format(np.var(clin_migr_cr[4])), end='\n\n')

# 3.- U-Mann Whitney
stat, pval = stats.mannwhitneyu(clin_migr_ep[4], clin_migr_cr[4], method="auto")
print("U-Mann Whitney statistic: {:.2f}".format(stat)+"; U-Mann Whitney p-value: {:.2f}".format(pval), end='\n\n')

################### OTHER PLOTS ###################
print('#################################################################################')
print('##################################### PLOTS #####################################')
print('#################################################################################', end='\n\n')

# Prepare the data scatter plot
BAG = np.concatenate((BAG_migr_ep_cor, BAG_migr_cr_cor, BAG_controls_cor))
real_ages = np.concatenate((edad_migr_ep, edad_migr_cr, edad_controls_sel))
predicted_ages = np.concatenate((pred_avg_migr_ep_cor, pred_avg_migr_cr_cor, pred_avg_controls_cor))
type_bag = np.concatenate((np.repeat('migr_ep', len(BAG_migr_ep_cor)), np.repeat('migr_cr', len(BAG_migr_cr_cor)),  np.repeat('controls', len(BAG_controls_cor))))
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
BAG = np.concatenate((BAG_migr_ep_cor, BAG_migr_cr_cor, BAG_controls_cor))
type_bag = np.concatenate((np.repeat('migr_ep', len(BAG_migr_ep_cor)), np.repeat('migr_cr', len(BAG_migr_cr_cor)),  np.repeat('controls', len(BAG_controls_cor))))

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
x = clin_migr_cr[2]
y = clin_migr_cr[1]
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#FF5555', alpha=1, marker="o")
plt.plot(x, poly1d_fn(x), color='#FF5555', linewidth=2)

x = clin_migr_ep[2]
y = clin_migr_ep[1]
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#00CCFF', alpha=1, marker="o")
plt.plot(x, poly1d_fn(x), color='#00CCFF', linewidth=2)

x = np.concatenate((clin_migr_ep[2],  clin_migr_cr[2]))
y = np.concatenate((clin_migr_ep[1],  clin_migr_cr[1]))
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.plot(x, poly1d_fn(x), color='#FFDD55', linewidth=2)

plt.xlabel('Migraine duration (years)', fontweight='bold')
plt.ylabel('BrainAGE (years)', fontweight='bold')
plt.show()


# Headache Frequency
x = clin_migr_cr[3]
y = clin_migr_cr[1]
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#FF5555', alpha=1, marker="o")
plt.plot(x, poly1d_fn(x), color='#FF5555', linewidth=2)

x = clin_migr_ep[3]
y = clin_migr_ep[1]
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#00CCFF', alpha=1, marker="o")
plt.plot(x, poly1d_fn(x), color='#00CCFF', linewidth=2)

x = np.concatenate((clin_migr_ep[3],  clin_migr_cr[3]))
y = np.concatenate((clin_migr_ep[1],  clin_migr_cr[1]))
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.plot(x, poly1d_fn(x), color='#FFDD55', linewidth=2)

plt.xlabel('Headache frequency (bouts/month)', fontweight='bold')
plt.ylabel('BrainAGE (years)', fontweight='bold')
plt.show()

# Migraine Frequency
x = clin_migr_cr[4]
y = clin_migr_cr[1]
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#FF5555', alpha=1, marker="o")
plt.plot(x, poly1d_fn(x), color='#FF5555', linewidth=2)

x = clin_migr_ep[4]
y = clin_migr_ep[1]
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#00CCFF', alpha=1, marker="o")
plt.plot(x, poly1d_fn(x), color='#00CCFF', linewidth=2)

x = np.concatenate((clin_migr_ep[4],  clin_migr_cr[4]))
y = np.concatenate((clin_migr_ep[1],  clin_migr_cr[1]))
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.plot(x, poly1d_fn(x), color='#FFDD55', linewidth=2)

plt.xlabel('Migraine frequency (bouts/month)', fontweight='bold')
plt.ylabel('BrainAGE (years)', fontweight='bold')
plt.show()

# Chronic migraine duration
x = clin_migr_cr[5]
y = clin_migr_cr[1]
coef = np.polyfit(x, y, 1)
poly1d_fn = np.poly1d(coef)
plt.scatter(x, y, color='#FF5555', alpha=1, marker="o")
plt.plot(x, poly1d_fn(x), color='#FF5555', linewidth=2)
plt.xlabel('Chronic migraine duration (months)', fontweight='bold')
plt.ylabel('BrainAGE (years)', fontweight='bold')
plt.show()

print("#################### END SCRIPT ####################")
print("####################################################")