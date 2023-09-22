import pingouin as pg

from Utils import *

from scipy.stats import pearsonr
from statsmodels.sandbox.stats.multicomp import multipletests

def plot_heatmap(matrix, features, comparisons):

    fig, ax = plt.subplots()
    heatmap = ax.imshow(matrix, cmap='coolwarm')
    cbar = ax.figure.colorbar(heatmap, ax=ax)
    cbar.ax.set_ylabel('Value', rotation=-90, va="bottom")

    ax.set_xticks(np.arange(len(matrix[0])))
    ax.set_yticks(np.arange(len(matrix)))
    ax.tick_params(labelsize=6)
    ax.tick_params(labelsize=6)
    ax.set_xticklabels(comparisons, rotation=45)
    ax.set_yticklabels(features)
    plt.show()

# config parser llamo al archivo de configuración
config_parser = configparser.ConfigParser(allow_no_value=True)
bindir = os.path.abspath(os.path.dirname(__file__))
config_parser.read(bindir + "/cfg.cnf")

folderApplication = config_parser.get("DATA", "DataAplication")


print('###########################################################################')
print('[INFO] Relationship between Features and and clinical characteristics...')
print('###########################################################################', end='\n\n')

combined_features = ['Mean_rh_wg_pct_precentral','GrayVol_lh_superiorfrontal','Mean_lh_wg_pct_precentral','normStdDev_Left-Caudate',
                    'Mean_lh_wg_pct_paracentral','normMax_Right-Pallidum','ThickAvg_rh_parsopercularis','normMax_Right-Caudate','normMax_Left-Pallidum',
                    'ThickAvg_lh_parstriangularis','normMax_3rd-Ventricle','Volume_mm3_Left-Putamen','normStdDev_Left-Cerebellum-Cortex',
                    'normMaxwm-lh-parsorbitalis', 'GrayVol_lh_lateralorbitofrontal','ThickAvg_lh_insula']

morphological_features = ['Volume_mm3_Right-Putamen', 'GrayVol_lh_superiorfrontal', 'GrayVol_lh_lateralorbitofrontal',
                        'SubCortGrayVol', 'ThickStd_rh_superiorfrontal', 'MeanCurv_rh_transversetemporal', 'ThickAvg_rh_precentral',
                        'GrayVol_rh_superiorfrontal', 'TotalGrayVol', 'GrayVol_rh_middletemporal', 'lh_MeanThickness', 'FoldInd_rh_rostralmiddlefrontal',
                        'Volume_mm3_Left-Caudate', 'GrayVol_rh_supramarginal', 'GrayVol_rh_insula', 'ThickAvg_lh_parsopercularis', 'ThickAvg_lh_precentral']

intensity_features = ['normStdDev_Left-Caudate', 'Mean_rh_wg_pct_precentral', 'Mean_lh_wg_pct_parsopercularis', 'normMax_Left-Pallidum',
                     'normMax_Right-Pallidum', 'Mean_lh_wg_pct_paracentral', 'normMax_Right-Caudate', 'normMaxwm-rh-insula',
                     'Mean_lh_wg_pct_precentral', 'normMin_Right-Cerebellum-Cortex', 'Mean_lh_wg_pct_superiorfrontal', 'normMax_CC_Central',
                     'Mean_rh_wg_pct_parsopercularis', 'normMax_Left-Caudate', 'normMax_3rd-Ventricle', 'normMinwm-lh-rostralmiddlefrontal',
                     'normMaxwm-lh-parstriangularis']

features_all = list(set(combined_features + morphological_features + intensity_features))

with open(os.path.join(folderApplication, 'Demographics_and_Clinical/featuresCorrelation_data.pkl'), 'rb') as f:
    data_CM, data_EM = pickle.load(f)

pval_freq_mig = []
# Freq migraine plots
for feature in features_all:

    x = np.array(data_CM['freq_mig']).astype(int)
    y = data_CM[feature].values

    # Calculate the best-fit line (linear regression)
    slope, intercept = np.polyfit(x, y, 1)
    regression_line = slope * x + intercept

    # Calculate the correlation coefficient and the p-value
    correlation_coef, p_value = pearsonr(x, y)
    pval_freq_mig.append(p_value)

    # Create a scatter plot
    plt.scatter(x, y, c='blue', label='Data Points')

    # Plot the regression line
    plt.plot(x, regression_line, c='red', label=f'Correlation Coef: {correlation_coef:.2f}\nP-value: {p_value:.4f}')

    # Add title and labels
    plt.title(feature+' vs '+'Migraine Frequency', fontweight='bold')
    plt.ylabel(feature)
    plt.xlabel('Migraine frequency')
    plt.legend()
    # plt.show()

# Apply Bonferroni correction
alpha = 0.05
reject, pval_freq_mig_corrected, _, _ = multipletests(pval_freq_mig, alpha=alpha, method='fdr_bh')

pval_tiempo_mig = []
for feature in features_all:

    x = np.array(data_CM['tiempo_mig']).astype(int)
    y = data_CM[feature].values

    # Calculate the best-fit line (linear regression)
    slope, intercept = np.polyfit(x, y, 1)
    regression_line = slope * x + intercept

    # Calculate the correlation coefficient and the p-value
    correlation_coef_p, p_value_p = pearsonr(x, y)

    # Define a dictionary containing employee data
    data = {'x': x, 'y': y,'cv': data_CM['Age'].values}
    df = pd.DataFrame(data)
    result_pg = pg.partial_corr(data=df, x='x', y='y', covar='cv').round(3)

    p_value = p_value_p
    p_value = result_pg['p-val'].values[0]
    correlation_coef = result_pg['r'].values[0]
    pval_tiempo_mig.append(p_value)

    # Create a scatter plot
    plt.scatter(x, y, c='blue', label='Data Points')

    # Plot the regression line
    plt.plot(x, regression_line, c='red', label=f'Correlation Coef 1: {correlation_coef_p:.2f}\nP-value: {p_value_p:.4f} \n Correlation Coef 2: {correlation_coef:.2f}\nP-value 2: {p_value:.4f} \n ')

    # Add title and labels
    plt.title(feature+' vs '+'Migraine duration (years)')
    plt.ylabel(feature)
    plt.xlabel('Migraine duration (years)')
    plt.legend()
    # plt.show()

# Apply Bonferroni correction
alpha = 0.05
reject, pval_tiempo_mig_corrected, _, _ = multipletests(pval_tiempo_mig, alpha=alpha, method='fdr_bh')

pval_freq_cef = []
for feature in features_all:

    x = np.array(data_CM['freq_cef']).astype(int)
    y = data_CM[feature].values

    # Calculate the best-fit line (linear regression)
    slope, intercept = np.polyfit(x, y, 1)
    regression_line = slope * x + intercept

    # Calculate the correlation coefficient and the p-value
    correlation_coef, p_value = pearsonr(x, y)
    pval_freq_cef.append(p_value)

    # Create a scatter plot
    plt.scatter(x, y, c='blue', label='Data Points')

    # Plot the regression line
    plt.plot(x, regression_line, c='red', label=f'Correlation Coef: {correlation_coef:.2f}\nP-value: {p_value:.4f}')

    # Add title and labels
    plt.title(feature+' vs '+'Hedache frequency')
    plt.ylabel(feature)
    plt.xlabel('Headache frequency')
    plt.legend()
    # plt.show()

# Apply Bonferroni correction
alpha = 0.05
reject, pval_freq_cef_corrected, _, _ = multipletests(pval_freq_cef, alpha=alpha, method='fdr_bh')

pval_tiempo_mig_cro = []
for feature in features_all:

    x = np.array(data_CM['tiempo_mig_cro']).astype(int)
    y = data_CM[feature].values

    # Calculate the best-fit line (linear regression)
    slope, intercept = np.polyfit(x, y, 1)
    regression_line = slope * x + intercept

    # Calculate the correlation coefficient and the p-value
    correlation_coef, p_value = pearsonr(x, y)

    # Define a dictionary containing employee data
    data = {'x': x, 'y': y,'cv': data_CM['Age'].values*12}
    df = pd.DataFrame(data)
    result_pg = pg.partial_corr(data=df, x='x', y='y', covar='cv').round(3)

    p_value = result_pg['p-val'].values[0]
    correlation_coef = result_pg['r'].values[0]
    pval_tiempo_mig_cro.append(p_value)

    # Create a scatter plot
    plt.scatter(x, y, c='blue', label='Data Points')

    # Plot the regression line
    plt.plot(x, regression_line, c='red', label=f'Correlation Coef 1: {correlation_coef_p:.2f}\nP-value: {p_value_p:.4f} \n Correlation Coef 2: {correlation_coef:.2f}\nP-value 2: {p_value:.4f} \n ')

    # Add title and labels
    plt.title(feature+' vs '+'Chronic migraine duration (months)')
    plt.ylabel(feature)
    plt.xlabel('Chronic migraine duration (months)')
    plt.legend()
    # plt.show()

# Apply Bonferroni correction
alpha = 0.05
reject, pval_tiempo_mig_cro_corrected, _, _ = multipletests(pval_tiempo_mig_cro, alpha=alpha, method='fdr_bh')

matrix = np.array([pval_tiempo_mig_corrected, pval_freq_mig_corrected, pval_freq_cef_corrected, pval_tiempo_mig_cro_corrected])
Four_comparisions = ["Migraine duration (years)", "Migraine frequency", "Headache frequency", "Chronic migraine duration (months)"]
plot_heatmap(matrix.transpose(), features_all, Four_comparisions)

pval_freq_mig = []
# Freq migraine plots
for feature in features_all:

    x = np.array(data_EM['freq_mig']).astype(int)
    y = data_EM[feature].values

    # Calculate the best-fit line (linear regression)
    slope, intercept = np.polyfit(x, y, 1)
    regression_line = slope * x + intercept

    # Calculate the correlation coefficient and the p-value
    correlation_coef, p_value = pearsonr(x, y)
    pval_freq_mig.append(p_value)

    # Create a scatter plot
    plt.scatter(x, y, c='blue', label='Data Points')

    # Plot the regression line
    plt.plot(x, regression_line, c='red', label=f'Correlation Coef: {correlation_coef:.2f}\nP-value: {p_value:.4f}')

    # Add title and labels
    plt.title(feature+' vs '+'Migraine Frequency', fontweight='bold')
    plt.ylabel(feature)
    plt.xlabel('Migraine frequency')
    plt.legend()
    # plt.show()

# Apply Bonferroni correction
alpha = 0.05
reject, pval_freq_mig_corrected, _, _ = multipletests(pval_freq_mig, alpha=alpha, method='fdr_bh')

pval_tiempo_mig = []
for feature in features_all:

    x = np.array(data_EM['tiempo_mig']).astype(int)
    y = data_EM[feature].values

    # Calculate the best-fit line (linear regression)
    slope, intercept = np.polyfit(x, y, 1)
    regression_line = slope * x + intercept

    # Calculate the correlation coefficient and the p-value
    correlation_coef_p, p_value_p = pearsonr(x, y)

    # Define a dictionary containing employee data
    data = {'x': x, 'y': y,'cv': data_EM['Age'].values}
    df = pd.DataFrame(data)
    result_pg = pg.partial_corr(data=df, x='x', y='y', covar='cv').round(3)

    p_value = result_pg['p-val'].values[0]
    correlation_coef = result_pg['r'].values[0]
    pval_tiempo_mig.append(p_value)

    # Create a scatter plot
    plt.scatter(x, y, c='blue', label='Data Points')

    # Plot the regression line
    plt.plot(x, regression_line, c='red', label=f'Correlation Coef 1: {correlation_coef_p:.2f}\nP-value: {p_value_p:.4f} \n Correlation Coef 2: {correlation_coef:.2f}\nP-value 2: {p_value:.4f} \n ')

    # Add title and labels
    plt.title(feature+' vs '+'Migraine duration (years)')
    plt.ylabel(feature)
    plt.xlabel('Migraine duration (years)')
    plt.legend()
    # plt.show()

# Apply Bonferroni correction
alpha = 0.05
reject, pval_tiempo_mig_corrected, _, _ = multipletests(pval_tiempo_mig, alpha=alpha, method='fdr_bh')

pval_freq_cef = []
for feature in features_all:

    x = np.array(data_EM['freq_cef']).astype(int)
    y = data_EM[feature].values

    # Calculate the best-fit line (linear regression)
    slope, intercept = np.polyfit(x, y, 1)
    regression_line = slope * x + intercept

    # Calculate the correlation coefficient and the p-value
    correlation_coef, p_value = pearsonr(x, y)
    pval_freq_cef.append(p_value)

    # Create a scatter plot
    plt.scatter(x, y, c='blue', label='Data Points')

    # Plot the regression line
    plt.plot(x, regression_line, c='red', label=f'Correlation Coef: {correlation_coef:.2f}\nP-value: {p_value:.4f}')

    # Add title and labels
    plt.title(feature+' vs '+'Hedache frequency')
    plt.ylabel(feature)
    plt.xlabel('Headache frequency')
    plt.legend()
    # plt.show()

# Apply Bonferroni correction
alpha = 0.05
reject, pval_freq_cef_corrected, _, _ = multipletests(pval_freq_cef, alpha=alpha, method='fdr_bh')

matrix = np.array([pval_tiempo_mig_corrected, pval_freq_mig_corrected, pval_freq_cef_corrected])
three_comparisions = ["Migraine duration (years)", "Migraine frequency", "Headache frequency"]
plot_heatmap(matrix.transpose(), features_all, three_comparisions)

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