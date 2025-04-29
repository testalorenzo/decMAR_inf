#
# Real-data analysis of iBeacon data
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

    #
    # Preprocessing
    #

    # Data can be downloaded from kaggle:
    # https://www.kaggle.com/datasets/mehdimka/ble-rssi-dataset/data

    # Load the data
    labeled = pd.read_csv('iBeacon_RSSI_Labeled.csv')
    unlabeled = pd.read_csv('iBeacon_RSSI_Unlabeled.csv')

    # Remove date columns 
    labeled.drop('date', axis=1, inplace=True)
    unlabeled.drop('date', axis=1, inplace=True)

    # format response variable to make it bivariate
    labeled['longitude'] = labeled['location'].apply(lambda x: x[0]) # -
    labeled['latitude'] = labeled['location'].apply(lambda x: int(x[1:3])) # |

    # Replace longitude with numeric values (ordered with alphabetical order)
    labeled['longitude'] = labeled['longitude'].apply(lambda x: ord(x))

    # Remove location column
    labeled = labeled.drop(columns=['location'])
    unlabeled = unlabeled.drop(columns=['location'])

    # Merge the two datasets
    data = pd.concat([labeled, unlabeled], axis=0)

    data['observed'] = 1
    data.loc[data['longitude'].isna(),'observed'] = 0

    # Format data for analysis

    X = labeled.drop(columns=['longitude', 'latitude']).to_numpy()
    Y = labeled[['longitude', 'latitude']].to_numpy()
    X_un = unlabeled.to_numpy()

    full_X = data.drop(columns=['longitude', 'latitude', 'observed']).to_numpy()
    full_R = data['observed'].to_numpy()
    full_Y = data[['longitude', 'latitude']].to_numpy()
    full_Y = np.nan_to_num(full_Y, nan=0)

    #
    # Modeling
    #

    # Fit random forest model Y ~ X
    mu_hat = np.zeros_like(full_Y)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(full_X):
        reg = RandomForestRegressor(random_state=4).fit(full_X[train_idx][full_R[train_idx] == 1], full_Y[train_idx][full_R[train_idx] == 1])
        # reg = LinearRegression().fit(predictors[train_idx][full_R[train_idx] == 1], full_Y[train_idx][full_R[train_idx] == 1])
        mu_hat[test_idx] = reg.predict(full_X[test_idx])

    # Fit offset logistic regression model R ~ X
    pi_hat = np.zeros(full_R.shape)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(full_R):
        clf = LogisticRegressionWithOffset().fit(R=full_R[train_idx], X=full_X[train_idx], offset=sum(full_R[train_idx]) / len(full_R[train_idx]))
        pi_hat[test_idx] = clf.predict_proba(full_X[test_idx])[:, 1]
    pi_hat = np.clip(pi_hat, 1e-6, 1)  # avoid division by zero

    # Fit AIPW estimator
    n_total = full_X.shape[0]
    residual = (full_Y - mu_hat) * (full_R[:, None] / pi_hat[:, None])
    aipw = mu_hat + residual.reshape((n_total, mu_hat.shape[1]))
    # theta_hat = LinearRegression(fit_intercept=False).fit(full_X, aipw).coef_
    # np.linalg.inv(full_X.T @ full_X) @ (full_X.T @ aipw) # equivalent to this
    theta_hat = aipw.mean(axis=0)
    theta_cov = aipw.T @ aipw / n_total # np.cov(before_sum)
    se = np.sqrt(np.diag(theta_cov) / n_total)

    # Fit naive estimator
    n = sum(full_R)
    naive = Y.mean(axis=0)
    naive_cov = Y.T @ Y / n # np.cov(before_sum_naive)
    se_naive = np.sqrt(np.diag(naive_cov) / n)

    #
    # Plotting
    #

    sns.set_theme(style="whitegrid", palette="pastel", font_scale=0.7)
    plt.rcParams["axes.grid"] = False

    width = 425

    axd = plt.figure(figsize=set_size(width, subplots=(1,2))).subplot_mosaic(
        [['pi', 'mu']],
        width_ratios=[0.3, 0.7],
        gridspec_kw = {'wspace':0.25, 'hspace':0.05}
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
    df = pd.DataFrame(Y)
    df['model'] = 'Y'
    df = pd.concat([df, pd.DataFrame(mu_hat)], axis=0)
    df['model'] = df['model'].replace({'Y': '$Y$', np.nan: '$\hat{\mu}$'})
    df.columns = ['longitude', 'latitude', 'model']

    sns.kdeplot(df, x='longitude', y='latitude', ax=axd['mu'], alpha=0.5, hue='model', palette=[palette[3], palette[0]], levels=10)
    # add dot for avg(Y)
    axd['mu'].scatter(naive[0], naive[1], color=palette[3], marker='o', s=20, label='Y')
    # add dot for theta_hat
    axd['mu'].scatter(theta_hat[0], theta_hat[1], color=palette[0], marker='o', s=20, label='$\hat{\mu}$')
    axd['mu'].set_xlabel('Longitude')
    axd['mu'].set_ylabel('Latitude')
    axd['mu'].set_title('Distribution of $Y$ and $\hat\mu$')
    axd['mu'].legend(loc="lower left")
    axd['mu'].tick_params(pad=-3)

    # add small grid to the plot
    axd['mu'].grid(which='both', color='gray', linestyle='--', linewidth=0.4)
    # letters instead of numbers for x axis
    axd['mu'].set_xticklabels(['0'] + [chr(i) for i in range(ord('A'), ord('Z') + 1, 5)])

    # Add error bars manually
    plt.savefig('app_ibeacon.pdf', bbox_inches='tight')