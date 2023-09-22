import re
import shap

from statsmodels.stats.multitest import multipletests
import scikit_posthocs as sp

from scipy import stats
from Utils import *

import matplotlib.pyplot as plt
import matplotlib as mpl

def formalize_feature_names(features):
    formalized_features = []
    for feature in features:
        words = re.split('-|_', feature)
        formalized_feature = ' '.join(words)
        formalized_features.append(formalized_feature)
    return formalized_features

def plot_heatmap(matrix, features):
    three_comparisions = ["HC-CM", "HC-EM", "EM-CM"]

    fig, ax = plt.subplots()
    heatmap = ax.imshow(matrix, cmap='viridis')
    cbar = ax.figure.colorbar(heatmap, ax=ax)
    cbar.ax.set_ylabel('Value', rotation=-90, va="bottom")

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            if matrix[i][j] < 0.001:
                label = "< 0.001 ***"
            elif matrix[i][j] < 0.01:
                label = "< 0.01 **"
            elif matrix[i][j] < 0.05:
                label = "< 0.05 *"
            elif matrix[i][j] > 0.05:
                label = str(float('%.2f' % matrix[i][j]))

            if matrix[i][j] < 0.1:
                ax.text(j, i, label, ha="center", va="center", color="white", size=10)
            else:
                ax.text(j, i, label, ha="center", va="center", color="black", size=10)

    ax.set_xticks(np.arange(len(matrix[0])))
    ax.set_yticks(np.arange(len(matrix)))
    ax.tick_params(labelsize=12)
    ax.tick_params(labelsize=12)
    ax.set_xticklabels(features, rotation=45)
    ax.set_yticklabels(three_comparisions)
    plt.show()

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
model = 'MLP40_Combined' # MLP40_Intensity; MLP40_Morphological;
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
covar_EM = load_list(os.path.join(folderApplication, 'Demographics_and_Clinical', 'covariates_EM.pkl'))
age_EM = covar_EM[0]
sex_EM = covar_EM[1]
etiv_EM = covar_EM[2]

covar_CM = load_list(os.path.join(folderApplication, 'Demographics_and_Clinical', 'covariates_CM.pkl'))
age_CM = covar_CM[0]
sex_CM = covar_CM[1]
etiv_CM = covar_CM[2]

covar_HC = load_list(os.path.join(folderApplication, 'Demographics_and_Clinical', 'covariates_HC.pkl'))
age_HC = covar_HC[0]
sex_HC = covar_HC[1]
etiv_HC = covar_HC[2]

# for each folder, get the file names sorted
model_list = sorted([model for model in os.listdir(folder_models) if 'MLP_nfeats_'+str(n_features) in model])

# load the validation ages by fold, required for bias correction
age_val_folds = load_list(os.path.join(folderCreation, 'BiasCorrectionData', 'age_val_folds_'+model_type.lower()+'.pkl'))
pred_val_folds = load_list(os.path.join(folderCreation, 'BiasCorrectionData', 'pred_val_folds_'+model_type.lower()+'.pkl'))

# load normalized controls, episodic migraine data and crhonic migraine data
HC_data = load_list(os.path.join(folderApplication, 'DATA', 'HC_data_norm_'+model_type.lower()+'.pkl'))
EM_data = load_list(os.path.join(folderApplication, 'DATA', 'EM_data_norm_'+model_type.lower()+'.pkl'))
CM_data = load_list(os.path.join(folderApplication, 'DATA', 'CM_data_norm_'+model_type.lower()+'.pkl'))

print('[## INFO ##] Finished Loading.', end='\n\n')

pred_HC, pred_CM, pred_EM = [], [], []
print("[## INFO ##] Calculating results by split...")
pred_HC = [pickle.load(open(os.path.join(folder_models, model_list[i]), 'rb')).predict(HC_data[i]) for i in range(10)]
pred_CM = [pickle.load(open(os.path.join(folder_models, model_list[i]), 'rb')).predict(CM_data[i]) for i in range(10)]
pred_EM = [pickle.load(open(os.path.join(folder_models, model_list[i]), 'rb')).predict(EM_data[i]) for i in range(10)]
print("[## INFO ##] Calculated.", end='\n\n')

pred_avg_HC = sum(pred_HC)/10
pred_avg_CM = sum(pred_CM)/10
pred_avg_EM = sum(pred_EM)/10

# load the SHAP value data
shap_folder = config_parser.get("SHAP", "shap_values_folder")
data = load_list(os.path.join(shap_folder, 'lista_shap_values_10_models_'+model_type+'.pkl'))

# divide data
expected_value = sum(data[3])/len(data[3])
data_HC = data[4][0][0]
data_EM = data[4][0][1]
data_CM = data[4][0][2]
shap_values_HC = data[0][0]
shap_EM = data[1][0]
shap_CM = data[2][0]

# save the SHAP values of each model of the ensembel in a dataframe. It is done form each group.
shap_values_HC_df = pd.DataFrame(np.concatenate(data[0], axis=1), columns=sum([df[0].columns.tolist() for df in data[4]], []))
shap_EM_df = pd.DataFrame(np.concatenate(data[1], axis=1), columns=sum([df[0].columns.tolist() for df in data[4]], []))
shap_CM_df = pd.DataFrame(np.concatenate(data[2], axis=1), columns=sum([df[0].columns.tolist() for df in data[4]], []))

HC_data_df = pd.concat([value[0] for value in data[4]], axis=1)
EM_data_df = pd.concat([value[1] for value in data[4]], axis=1)
CM_data_df = pd.concat([value[2] for value in data[4]], axis=1)

# calculate the shap value for each feature in each of the groups.
dictionary = {'shap_vals':sum(abs(np.concatenate(data[0], axis=1)))/82, 'features': sum([df[0].columns.tolist() for df in data[4]], [])}
HC_df = pd.DataFrame(dictionary)
HC_df.sort_values(by='shap_vals', ascending=False, inplace=True)

dictionary = {'shap_vals':sum(abs(np.concatenate(data[1], axis=1)))/91, 'features': sum([df[1].columns.tolist() for df in data[4]], [])}
EM_df = pd.DataFrame(dictionary)
EM_df.sort_values(by='shap_vals', ascending=False, inplace=True)

dictionary = {'shap_vals':sum(abs(np.concatenate(data[2], axis=1)))/74, 'features': sum([df[2].columns.tolist() for df in data[4]], [])}
CM_df = pd.DataFrame(dictionary)
CM_df.sort_values(by='shap_vals', ascending=False, inplace=True)

# Rename and erase the repeated features
features = [feature[:-8] for feature in HC_df['features'].tolist()]
HC_df['renamed_features'] = EM_df['renamed_features'] = CM_df['renamed_features'] = features
unique_features = list(set(features))

print('[INFO] Calculate the SHAP value for each feature:')
list_aux_HC_shap, list_aux_EM_shap, list_aux_CM_shap, list_features_shap = [], [], [], []
for feature in unique_features:
    aux_HC, aux_EM, aux_CM = 0, 0, 0
    count = 0
    for model_feature in shap_values_HC_df.columns.tolist():
        if model_feature[:-8] == feature:
            aux_HC = aux_HC + shap_values_HC_df[model_feature].values
            aux_EM = aux_EM + shap_EM_df[model_feature].values
            aux_CM = aux_CM + shap_CM_df[model_feature].values
            count = count+1
    list_aux_HC_shap.append(aux_HC)
    list_aux_EM_shap.append(aux_EM)
    list_aux_CM_shap.append(aux_CM)
    list_features_shap.append(feature)

shap_values_HC_df = pd.DataFrame(np.transpose(np.vstack(list_aux_HC_shap)), columns=list_features_shap)
shap_EM_df = pd.DataFrame(np.transpose(np.vstack(list_aux_EM_shap)), columns=list_features_shap)
shap_CM_df = pd.DataFrame(np.transpose(np.vstack(list_aux_CM_shap)), columns=list_features_shap)
print('[INFO] Calculated.', end='\n\n')

print('[INFO] Calculate the value of the feature as an average among the ten folds:')
list_aux_HC_feat, list_aux_EM_feat, list_aux_CM_feat, list_features_feat = [], [], [], []
for feature in unique_features: # or thirty_features
    aux_HC, aux_EM, aux_CM = 0, 0, 0
    count = 0
    for model_feature in HC_data_df.columns.tolist():
        if model_feature[:-8] == feature:
            aux_HC = aux_HC + HC_data_df[model_feature].values
            aux_EM = aux_EM + EM_data_df[model_feature].values
            aux_CM = aux_CM + CM_data_df[model_feature].values
            count = count+1
    list_aux_HC_feat.append(aux_HC)
    list_aux_EM_feat.append(aux_EM)
    list_aux_CM_feat.append(aux_CM)
    list_features_feat.append(feature)

values_feat_HC_df = pd.DataFrame(np.transpose(np.vstack(list_aux_HC_feat)), columns=list_features_feat)
values_feat_EM_df = pd.DataFrame(np.transpose(np.vstack(list_aux_EM_feat)), columns=list_features_feat)
values_feat_CM_df = pd.DataFrame(np.transpose(np.vstack(list_aux_CM_feat)), columns=list_features_feat)
print('[INFO] Calculated.', end='\n\n')

print('[INFO] Calculate predictions with the SHAP model:')
Pred_HC_with_shap = sum(np.transpose(shap_values_HC_df.values))/10+expected_value
Pred_EM_with_shap = sum(np.transpose(shap_EM_df.values))/10+expected_value
Pred_CM_with_shap = sum(np.transpose(shap_CM_df.values))/10+expected_value
print('[INFO] Calculated.', end='\n\n')

print("Error (MAE) between the predicted age with the MLP and the SHAP model: ")
print('Controls: {:.2f}'.format(mean_absolute_error(pred_avg_HC, Pred_HC_with_shap)))
print('Episodic migraine: {:.2f}'.format(mean_absolute_error(pred_avg_EM, Pred_EM_with_shap)))
print('Chronic migraine: {:.2f}'.format(mean_absolute_error(pred_avg_CM, Pred_CM_with_shap)), end='\n\n')

# plots with shap library
# plot1 = shap.summary_plot(shap_values_HC_df.values, values_feat_HC_df, show=False)
# plot2 = shap.summary_plot(shap_EM_df.values, values_feat_EM_df, show=False)
# plot3 = shap.summary_plot(shap_CM_df.values, values_feat_CM_df, show=False)

all_shap_values = pd.concat([shap_values_HC_df, shap_EM_df, shap_CM_df])
all_values_shap = pd.concat([values_feat_HC_df, values_feat_EM_df, values_feat_CM_df])

# plots with shap library
# plot = shap.force_plot(expected_value, all_shap_values.values/10, all_values_shap, show=False)
# shap.save_html("index.htm", plot)

# Select the features not repeated among the fifteen 15 best of the three groups
# find the best fifteen for each group
HC_fifteen_best = (shap_values_HC_df/10).abs().sum().sort_values(ascending=False)[0:15].index.tolist()
EM_fifteen_best = (shap_EM_df/10).abs().sum().sort_values(ascending=False)[0:15].index.tolist()
CM_fifteen_best = (shap_CM_df/10).abs().sum().sort_values(ascending=False)[0:15].index.tolist()

# find the unrepeated ones
fifteen_feat_HC = list(set(HC_fifteen_best+EM_fifteen_best+CM_fifteen_best))
print('[INFO] Unrepeated features among the 15 best features of each of the groups:')
print(fifteen_feat_HC, end='\n\n')

# SHAP values for each of the features selected for each group
three_shap_vals_means = pd.concat([(shap_values_HC_df/10).abs().sum(), (shap_EM_df/10).abs().sum(), (shap_CM_df/10).abs().sum()], axis=1)
three_shap_vals_means = three_shap_vals_means.transpose()
three_shap_vals_means = three_shap_vals_means[fifteen_feat_HC]
three_shap_vals_means = three_shap_vals_means.transpose()
three_shap_vals_means.sort_values(by=0, ascending=False, inplace=True)
features = three_shap_vals_means.index.tolist()

print('######################################')
print('######## Kruskal-Wallis Tests ########')
print('######################################', end='\n\n')

kruskal_comparisons = []
for feature in features:
    stat, pval = stats.kruskal(shap_values_HC_df[feature].values/10, shap_EM_df[feature].values/10, shap_CM_df[feature].values/10)
    kruskal_comparisons.append(stats.kruskal(shap_values_HC_df[feature].values/10, shap_EM_df[feature].values/10, shap_CM_df[feature].values/10)[1])
    print('-------------------------------------')
    print('feature: ' + feature)
    print("Kruskal-Wallis 3 groups statistic: {:.2f}".format(stat) + "; Kruskal-Wallis 3 groups p-value: {:.2f}".format(pval))
    print('-------------------------------------', end='\n\n')

print("[INFO] Results before FDR correction. Apply correction. If still significative, Connover-Iman Test.")

bh_p_vals = multipletests(np.array(kruskal_comparisons), alpha=0.05, method='fdr_bh', is_sorted=False, returnsorted=False)

conover_matrices, sig_feature = [], []
for i in range(0, len(features), 1):
    feature = features[i]
    if bh_p_vals[1][i] < 0.05:
        print('Feature '+feature +' KW test was statistically significative, apply 2 vs 2 comparison:', end='\n\n')

        d = {'SHAP_VAL': np.concatenate((shap_values_HC_df[feature].values/10, shap_EM_df[feature].values/10, shap_CM_df[feature].values/10)) ,
             'Group': np.concatenate((np.repeat('HC', len(shap_values_HC_df[feature].values)), np.repeat('EM', len(shap_EM_df[feature].values/10)), np.repeat('CM', len(shap_CM_df[feature].values/10))))}
        df = pd.DataFrame(data=d)
        print(sp.posthoc_conover(df, val_col='SHAP_VAL', group_col='Group', p_adjust = 'fdr_bh'), end='\n\n')
        conover_matrices.append(sp.posthoc_conover(df, val_col='SHAP_VAL', group_col='Group', p_adjust = None))
        sig_feature.append(feature)

print('#################################################################################')
print('##################################### PLOTS #####################################')
print('#################################################################################', end='\n\n')

# bar plot importance
d = {'features':features , 'pvals':bh_p_vals[1]}
dataframe_kruskal = pd.DataFrame(data=d)
values = dataframe_kruskal['pvals'].values
features_formal = formalize_feature_names(dataframe_kruskal['features'].tolist())
fig, ax = plt.subplots()
cmap = mpl.colormaps.get_cmap('viridis')
colors = cmap(values)
ax.bar(range(len(values)), values, color=colors, width = 1)
ax.axhline(y=0.05, linestyle='--', color='red')
ax.set_xticks(range(len(values)))
ax.set_xticklabels([value for value in features_formal], rotation=45, ha='right')
ax.set_ylim([0, 1])
ax.set_xlabel('Features', fontsize=12)
ax.set_ylabel('Values', fontsize=12)
plt.show()

p_values = []
for i in range (0, len(conover_matrices), 1):
    matrix = conover_matrices[i]
    p_values.append(np.array((matrix.iloc[0, 2], matrix.iloc[1, 2], matrix.iloc[0, 1]))) # ORDER HCEM; HCCM; EMCM;

if len(sig_feature) != 0:
    sig_feature_formal = formalize_feature_names(sig_feature)
    plot_heatmap(np.array(np.transpose(p_values)), sig_feature_formal)
else:
    print('No statistically significant features.')

# Bar plot importance across groups
three_shap_vals_std = pd.concat([(shap_values_HC_df/10).abs().std(), (shap_EM_df/10).abs().std(), (shap_CM_df/10).abs().std()], axis=1)
three_shap_vals_std = three_shap_vals_std.transpose()
three_shap_vals_std = three_shap_vals_std[features]
three_shap_vals_std = three_shap_vals_std.transpose()

n_groups = len(features_formal)
means_controls = three_shap_vals_means.iloc[0:n_groups, 0].values
means_ep = three_shap_vals_means.iloc[0:n_groups, 1].values
means_cr = three_shap_vals_means.iloc[0:n_groups, 2].values

std_controls = three_shap_vals_std.iloc[0:n_groups, 0].values
std_ep = three_shap_vals_std.iloc[0:n_groups, 1].values
std_cr = three_shap_vals_std.iloc[0:n_groups, 2].values

# create plot
fig, ax = plt.subplots()
index = np.arange(n_groups)
bar_width = 0.25
opacity = 0.8
rects1 = plt.barh(index + bar_width, means_controls[::-1], bar_width, alpha=opacity, color='#00FF00', label='Controls')
rects2 = plt.barh(index, means_ep[::-1], bar_width, alpha=opacity, color='#00CCFF', label='EM')
rects3 = plt.barh(index - bar_width, means_cr[::-1], bar_width, alpha=opacity, color='#FF5555', label='CM')
plt.xlabel('Absolute sum of the SHAP values', fontweight="bold")
plt.ylabel('Features', fontweight="bold")
plt.yticks(index, tuple(list(features_formal[::-1])))
plt.legend()
plt.show()

print("#################### END SCRIPT ####################")
print("####################################################")

