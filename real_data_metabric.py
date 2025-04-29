#
# Real-data analysis of METABRIC
#

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestRegressor
from sklearn.base import BaseEstimator, ClassifierMixin
from statsmodels.api import GLM, families
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression, LogisticRegression

def set_size(width, fraction=1, subplots=(3, 3)):
    """Set figure dimensions to avoid scaling in LaTeX.

    Parameters
    ----------
    width: float or string
            Document width in points, or string of predined document type
    fraction: float, optional
            Fraction of the width which you wish the figure to occupy
    subplots: array-like, optional
            The number of rows and columns of subplots.
    Returns
    -------
    fig_dim: tuple
            Dimensions of figure in inches
    """
    if width == 'thesis':
        width_pt = 426.79135
    elif width == 'beamer':
        width_pt = 307.28987
    else:
        width_pt = width

    # Width of figure (in pts)
    fig_width_pt = width_pt * fraction
    # Convert from pt to inches
    inches_per_pt = 1 / 72.27

    # Golden ratio to set aesthetic figure height
    # https://disq.us/p/2940ij3
    golden_ratio = (5**.5 - 1) / 2

    # Figure width in inches
    fig_width_in = fig_width_pt * inches_per_pt
    # Figure height in inches
    fig_height_in = fig_width_in * golden_ratio * (subplots[0] / subplots[1])

    return (fig_width_in, fig_height_in)

class LogisticRegressionWithOffset(BaseEstimator, ClassifierMixin):
    def fit(self, X, R, offset=None):
        self.offset = offset
        self.fitted = GLM(R, X, family=families.Binomial(), offset=[self.offset]*X.shape[0]).fit()
        return self

    def predict_proba(self, X):
        self.offset = self.offset
        pr = self.fitted.predict(X, offset=[self.offset]*X.shape[0]).reshape(-1, 1)
        return np.concatenate([1 - pr, pr], axis=1)

    def predict(self, X):
        return 1*(self.predict_proba(X)[:,1]>0.5)

if __name__ == '__main__':

    full_analysis = False

    #
    # Preprocessing
    #

    # Data can be downloaded from cbioportal:
    # https://www.cbioportal.org/study/summary?id=brca_metabric

    # Load the data in txt file
    clinical_p = pd.read_csv('data_clinical_patient.txt', sep='\t', skiprows=4)
    clinical_s = pd.read_csv('data_clinical_sample.txt', sep='\t', skiprows=4)
    rna = pd.read_csv('data_mrna_illumina_microarray.txt', sep='\t')

    # Filter rna data picking only among the biomarkers of interest
    # ESR1, PGR, ERBB2, MKI67, PLAU, ELAVL1, EGFR, BTRC, FBXO6, SHMT2,
    # KRAS, SRPK2, YWHAQ, PDHA1, EWSR1, ZDHHC17, ENO1, DBN1, PLK1 and GSK3B

    biomarkers = ['ESR1', 'PGR', 'ERBB2', 'MKI67', 'PLAU', 'ELAVL1', 'EGFR',
                  'BTRC', 'FBXO6', 'SHMT2', 'KRAS', 'SRPK2', 'YWHAQ', 'PDHA1', 
                  'EWSR1', 'ZDHHC17', 'ENO1', 'DBN1', 'PLK1', 'GSK3B']
    
    # Filter the rna data
    rna = rna[rna['Hugo_Symbol'].isin(biomarkers)]
    rna = rna.set_index('Hugo_Symbol')
    rna = rna.drop(columns=['Entrez_Gene_Id'])
    rna = rna.transpose()

    samples = rna.index.tolist()

    # Filter the clinical data
    clinical_p = clinical_p[clinical_p['PATIENT_ID'].isin(samples)]
    clinical_s = clinical_s[clinical_s['SAMPLE_ID'].isin(samples)]

    # Merge clinical data
    clinical = pd.merge(clinical_p, clinical_s, on='PATIENT_ID', how='inner')
    clinical = clinical.set_index('PATIENT_ID')

    # Extract label
    R = clinical['VITAL_STATUS']

    # Extract response
    Y = clinical['OS_MONTHS']

    # Filter the clinical data
    # age, menopausal state, tumor size, radiotherapy, chemotherapy, hormone therapy,
    # neoplasm histologic grade, cellularity, surgery-breast conserving
    # and surgery-mastectomy
    covariates = ['AGE_AT_DIAGNOSIS', 'INFERRED_MENOPAUSAL_STATE', 'TUMOR_SIZE',
                  'RADIO_THERAPY', 'CHEMOTHERAPY',
                  'HORMONE_THERAPY', 'HISTOLOGICAL_SUBTYPE',
                  'CELLULARITY', 'BREAST_SURGERY']
    clinical = clinical[covariates]

    # Filter the clinical data to match rna
    clinical = clinical[clinical.index.isin(samples)]

    # Drop NA in clinical data
    clinical = clinical.dropna()

    # One-hot encode categorical variables
    clinical = pd.get_dummies(clinical, columns=['INFERRED_MENOPAUSAL_STATE',
                                                 'RADIO_THERAPY',
                                                 'CHEMOTHERAPY',
                                                 'HORMONE_THERAPY',
                                                 'HISTOLOGICAL_SUBTYPE',
                                                 'CELLULARITY',
                                                 'BREAST_SURGERY'], 
                                                drop_first=True)

    Y = Y[Y.index.isin(samples) & Y.index.isin(clinical.index)]
    R = R[R.index.isin(samples) & R.index.isin(clinical.index)]

    # Format label to make supervised/unsupervised analysis
    # 1: died of cancer; 0: otherwise
    if full_analysis:
        R = R.replace({'Died of Disease': 1,
                    'Died of Other Causes': 0,
                    'Living': 0})
    else:
        R = R.replace({'Died of Disease': 1,
                    'Died of Other Causes': 0,
                    'Living': 2})
        R = R[R != 2]
    R = R.dropna()

    # Create supervised datasets
    X = rna[rna.index.isin(R[R==1].index)]
    W = clinical[clinical.index.isin(R[R==1].index)]
    Y = Y[Y.index.isin(R[R==1].index)]

    X_un = rna[rna.index.isin(R[R==0].index)]
    W_un = clinical[clinical.index.isin(R[R==0].index)]

    # Order the data
    X = X.sort_index()
    W = W.sort_index()
    Y = Y.sort_index()
    R = R.sort_index()
    X_un = X_un.sort_index()
    W_un = W_un.sort_index()

    # Y = (Y - Y.mean()) / Y.std()

    full_X = pd.concat([X, X_un], axis=0)
    full_W = pd.concat([W, W_un], axis=0)
    zeros = pd.Series(np.zeros((full_X.shape[0] - Y.shape[0])), index=X_un.index)
    full_Y = pd.concat([Y, zeros], axis=0)

    full_X = full_X.sort_index().to_numpy()
    full_W = full_W.sort_index().to_numpy()
    full_Y = full_Y.sort_index().to_numpy()
    full_R = R.to_numpy()

    full_W = full_W.astype(np.float64)
    predictors = np.concatenate([full_X, full_W], axis=1)
    # predictors = (predictors - predictors.mean(axis=0)) / predictors.std(axis=0)

    #
    # Modeling
    #

    # Fit random forest model Y ~ X + W
    mu_hat = np.zeros_like(full_Y)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(predictors):
        reg = RandomForestRegressor(random_state=4).fit(predictors[train_idx][full_R[train_idx] == 1], full_Y[train_idx][full_R[train_idx] == 1])
        # reg = LinearRegression().fit(predictors[train_idx][full_R[train_idx] == 1], full_Y[train_idx][full_R[train_idx] == 1])
        mu_hat[test_idx] = reg.predict(predictors[test_idx])

    # Fit offset logistic regression model R ~ X + W
    pi_hat = np.zeros_like(full_Y)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(predictors):
        clf = LogisticRegression(max_iter=10000).fit(predictors[train_idx], full_R[train_idx])
        pi_hat[test_idx] = clf.predict_proba(predictors[test_idx])[:, 1]
    pi_hat = np.clip(pi_hat, 1e-6, 1)  # avoid division by zero

    # Fit AIPW estimator
    n_total = full_X.shape[0]
    residual = (full_Y - mu_hat)[:,None] * (full_R[:, None] / pi_hat[:, None])
    aipw = mu_hat + residual.reshape(n_total)
    # theta_hat = LinearRegression(fit_intercept=False).fit(full_X, aipw).coef_
    # np.linalg.inv(full_X.T @ full_X) @ (full_X.T @ aipw) # equivalent to this
    multiplier = np.linalg.inv(full_X.T @ full_X) @ full_X.T
    before_sum = (multiplier * aipw) * n_total
    theta_hat = before_sum.mean(axis=1)
    theta_cov = before_sum @ before_sum.T / n_total # np.cov(before_sum)
    se = np.sqrt(np.diag(theta_cov) / n_total)

    # Fit naive estimator
    n = sum(R)
    # naive = LinearRegression(fit_intercept=False).fit(full_X[full_R == 1], full_Y[full_R == 1]).coef_
    multiplier = np.linalg.inv(full_X[full_R == 1].T @ full_X[full_R == 1]) @ full_X[full_R == 1].T
    before_sum_naive = (multiplier * full_Y[full_R == 1]) * n
    naive = before_sum_naive.mean(axis=1)
    naive_cov = before_sum_naive @ before_sum_naive.T / n # np.cov(before_sum_naive)
    se_naive = np.sqrt(np.diag(naive_cov) / n)

    # Print significant biomarkers according to AIPW
    print('AIPW significant biomarkers:')
    for i, biomarker in enumerate(biomarkers):
        if abs(theta_hat[i]) > 1.645 * se[i]:
            print(f'{biomarker}: {theta_hat[i]:.3f} ({se[i]:.3f})')
    # Print significant biomarkers according to naive
    print('Naive significant biomarkers:')
    for i, biomarker in enumerate(biomarkers):
        if abs(naive[i]) > 1.645 * se_naive[i]:
            print(f'{biomarker}: {naive[i]:.3f} ({se_naive[i]:.3f})')

    sns.set_theme(style="whitegrid", palette="pastel", font_scale=0.7)
    plt.rcParams["axes.grid"] = False

    width = 425

    axd = plt.figure(figsize=set_size(width, subplots=(2,2))).subplot_mosaic(
        [['pi', 'mu'],
         ['coef','coef']],
        width_ratios=[1, 1],
        height_ratios=[1, 1],
        gridspec_kw = {'wspace':0.25, 'hspace':0.4}
    )

    palette = sns.color_palette("pastel")

    # Plot n/n_total and pi_hat distribution
    sns.histplot(pi_hat, ax=axd['pi'], alpha=0.5, label='$\hat{\pi}$', kde=True, color=palette[4])
    axd['pi'].axvline(np.mean(pi_hat), linestyle='--', color=palette[4])
    axd['pi'].axvline(np.mean(full_R), color='gray', linestyle='--')
    axd['pi'].set_xlabel('Value')
    axd['pi'].set_ylabel('Count')
    axd['pi'].set_title('Distribution of $\hat\pi$')
    axd['pi'].legend(loc="upper right")
    axd['pi'].tick_params(pad=-3)

    # Plot marginal of Y vs marginal of mu_hat
    sns.histplot(Y, ax=axd['mu'], alpha=0.5, label='Y', kde=True, color=palette[6])
    axd['mu'].axvline(np.mean(Y), linestyle='--', color=palette[6])
    sns.histplot(mu_hat, ax=axd['mu'], alpha=0.5, label='$\hat{\mu}$', kde=True, color=palette[9])
    axd['mu'].axvline(np.mean(mu_hat), linestyle='--', color=palette[9])
    axd['mu'].set_xlabel('Value')
    axd['mu'].set_ylabel('Count')
    axd['mu'].set_title('Distribution of $Y$ and $\hat\mu$')
    axd['mu'].legend(loc="upper right")
    axd['mu'].tick_params(pad=-3)

    # Prepare data for seaborn pointplot
    df_plot = pd.DataFrame({
        'Biomarker': biomarkers * 2,
        'Estimate': np.concatenate([theta_hat, naive]),
        'SE': np.concatenate([1.645 * se, 1.645 * se_naive]),
        'Method': ['DS$^3$'] * len(biomarkers) + ['Naive'] * len(biomarkers)
    })

    sns.pointplot(data=df_plot, x='Biomarker', y='Estimate', hue='Method', dodge=0.4,
                  palette={'DS$^3$': palette[0], 'Naive':palette[3]}, errorbar=None, ax=axd['coef'],
                  markersize=3, linestyle='none')

    # Add error bars manually
    for i, row in df_plot.iterrows():
        xloc = i % len(biomarkers) + (0.2 if row['Method'] == 'Naive' else -0.2)
        color = palette[3] if row['Method'] == 'Naive' else palette[0]
        axd['coef'].errorbar(xloc, row['Estimate'], yerr=row['SE'], fmt='none',
                    capsize=0, elinewidth=1, ecolor=color)
        
    axd['coef'].axhline(0, linestyle='--', color='gray')
    axd['coef'].set_xticks(range(len(biomarkers)))
    axd['coef'].set_xticklabels(biomarkers, rotation=90)
    axd['coef'].set_xlabel('Biomarkers')
    axd['coef'].set_ylabel('Effect size')
    axd['coef'].set_title('Estimated coefficients')
    axd['coef'].tick_params(pad=-3)

    sns.move_legend(axd['coef'], "upper right", ncol=2, title='')

    if full_analysis:
        plt.savefig('app.pdf', bbox_inches='tight')
    else:
        plt.savefig('app_competing.pdf', bbox_inches='tight')

    # Plot Y vs mu_hat1
    axd = plt.figure(figsize=set_size(width, subplots=(1,1))).subplot_mosaic(
        [['main']],
        gridspec_kw = {'wspace':0.05, 'hspace':0.05}
    )

    mu_hat1 = mu_hat[full_R == 1]

    df = pd.DataFrame({
        'Y': Y,
        'mu_hat': mu_hat1,
    })

    sns.scatterplot(data=df, x='Y', y='mu_hat', ax=axd['main'], color=palette[0])
    axd['main'].set_xlabel('Y')
    axd['main'].set_ylabel('$\hat\mu$')
    axd['main'].set_title('Joint distribution of Y and $\hat\mu$ in $R=1$ group')
    # add line y=x
    axd['main'].plot([20, 175], [20, 175], linestyle='--', color='gray')

    if full_analysis:
        plt.savefig('jointplot_Ymu.pdf', bbox_inches='tight')
    else:
        plt.savefig('jointplot_Ymu_competing.pdf', bbox_inches='tight')