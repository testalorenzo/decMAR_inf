# Plot simulation results
#

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Safe unpickler to fix numpy._core issue
class SafeUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core")
        return super().find_class(module, name)

if __name__ == "__main__":

    sns.set_theme(style="whitegrid", palette="pastel", font_scale=0.7)
    plt.rcParams["axes.grid"] = False

    width = 425

    axd = plt.figure(figsize=set_size(width, subplots=(1,2))).subplot_mosaic(
        [['mse', 'coverage']],
        width_ratios=[1, 1],
        gridspec_kw = {'wspace':0.25, 'hspace':0.05}
    )

    # Load the pickle file safely
    with open("simulation_results_lm.pkl", "rb") as f:
    # with open('simulation_results_mom_big.pkl', 'rb') as f:
    # with open("coef_simulation_results_lm.pkl", "rb") as f:
        results = SafeUnpickler(f).load()

    # Define the list of methods
    methods = ['aipw', 'or', 'ipw', 'naive']
    n = 100
    multiplier = 10
    N = n * multiplier
    setting = 'logistic'
    pi_model = 'logistic'

    # Initialize summary containers
    mse_summary = []
    coverage_summary = []

    mse_extended = []
    coverage_extended = []

    # Loop through each method and compute MSE and coverage
    for method in methods:
        method_mse = []
        method_cov = []

        for sim in results:
            if sim['n'] != n:
                continue
            if sim['N'] != N:
                continue
            if sim['setting'] != setting:
                continue
            if sim['pi_model'] != pi_model:
                continue

            est = np.array(sim['estimates'][method])
            true = np.array(sim['estimates']['true_theta'])
            cov = sim['estimates'][f"{method}_cov"]

            mse = np.mean((est.astype(float) - true.astype(float)) ** 2, axis=1)
            method_mse.append(mse)
            method_cov.append(cov)

        mse_extended.append(method_mse)
        coverage_extended.append(method_cov)

        mse_mean = np.mean(method_mse)
        mse_sd = np.std(method_mse)
        cov_mean = np.mean(method_cov)
        cov_sd = np.std(method_cov)

        mse_summary.append((method, mse_mean, mse_sd))
        coverage_summary.append((method, cov_mean, cov_sd))

    # Convert to DataFrames
    mse_df = pd.DataFrame(mse_summary, columns=['Method', 'Avg_MSE', 'SD_MSE'])
    cov_df = pd.DataFrame(coverage_summary, columns=['Method', 'Avg_Coverage', 'SD_Coverage'])

    # Prepare boxplot data (flatten if nested)
    mse_data = [np.array(m).flatten() for m in mse_extended]
    cov_data = [np.array(c).flatten() for c in coverage_extended]

    # Convert cov_data elements to float
    cov_data = [np.array(c, dtype=float) for c in cov_data]

    methods = ['AIPW', 'OR', 'IPW', 'Naive']

    plot_box = pd.DataFrame(mse_data).T
    plot_box.columns = methods

    # Plot the results

    # Boxplot MSE
    sns.boxplot(
        data=plot_box,
        ax=axd['mse'],
        palette="pastel",
        fliersize=3
    )

    axd['mse'].set_title('MSE Distribution')
    axd['mse'].set_xlabel('Method')
    axd['mse'].set_ylabel('MSE')
    axd['mse'].tick_params(pad=-3)

    # Calculate mean and standard error for coverage
    cov_means = [np.mean(c) for c in cov_data]
    cov_se = [np.std(c) / np.sqrt(len(c)) for c in cov_data]

    # Make covdata a dataframe
    cov_df = pd.DataFrame(cov_data).T
    cov_df.columns = methods

    # Coverage errorbar (point + bar)
    sns.pointplot(
        data=cov_df,
        ax=axd['coverage'],
        palette="pastel",
        errorbar=('se',1.96),
        capsize=0,
        markersize=3,
        err_kws={'linewidth': 1}
    )

    # add horizontal line for nominal coverage
    axd['coverage'].axhline(0.95, color='gray', linestyle='--', linewidth=1, label='Nominal 95%')
    axd['coverage'].set_ylim(-0.05, 1.05)
    axd['coverage'].set_title('Empirical Coverage')
    axd['coverage'].set_ylabel('Coverage')
    axd['coverage'].set_xlabel('Method')
    axd['coverage'].tick_params(pad=-3)

    plt.savefig('sim_mean.pdf', bbox_inches='tight')


    axd = plt.figure(figsize=set_size(width, subplots=(1,2))).subplot_mosaic(
        [['mse', 'coverage']],
        width_ratios=[1, 1],
        gridspec_kw = {'wspace':0.25, 'hspace':0.05}
    )

    # Load the pickle file safely
    with open("coef_simulation_results_lm.pkl", "rb") as f:
        results = SafeUnpickler(f).load()

    # Define the list of methods
    methods = ['aipw', 'or', 'ipw', 'naive']
    n = 100
    multiplier = 10
    N = n * multiplier
    setting = 'logistic'
    pi_model = 'logistic'

    # Initialize summary containers
    mse_summary = []
    coverage_summary = []

    mse_extended = []
    coverage_extended = []

    # Loop through each method and compute MSE and coverage
    for method in methods:
        method_mse = []
        method_cov = []

        for sim in results:
            if sim['n'] != n:
                continue
            if sim['N'] != N:
                continue
            if sim['setting'] != setting:
                continue
            if sim['pi_model'] != pi_model:
                continue

            est = np.array(sim['estimates'][method])
            true = np.array(sim['estimates']['true_theta'])
            cov = sim['estimates'][f"{method}_cov"]

            mse = np.mean((est.astype(float) - true.astype(float)) ** 2, axis=1)
            method_mse.append(mse)
            method_cov.append(cov)

        mse_extended.append(method_mse)
        coverage_extended.append(method_cov)

        mse_mean = np.mean(method_mse)
        mse_sd = np.std(method_mse)
        cov_mean = np.mean(method_cov)
        cov_sd = np.std(method_cov)

        mse_summary.append((method, mse_mean, mse_sd))
        coverage_summary.append((method, cov_mean, cov_sd))

    # Convert to DataFrames
    mse_df = pd.DataFrame(mse_summary, columns=['Method', 'Avg_MSE', 'SD_MSE'])
    cov_df = pd.DataFrame(coverage_summary, columns=['Method', 'Avg_Coverage', 'SD_Coverage'])

    # Prepare boxplot data (flatten if nested)
    mse_data = [np.array(m).flatten() for m in mse_extended]
    cov_data = [np.array(c).flatten() for c in coverage_extended]

    # Convert cov_data elements to float
    cov_data = [np.array(c, dtype=float) for c in cov_data]

    methods = ['AIPW', 'OR', 'IPW', 'Naive']

    plot_box = pd.DataFrame(mse_data).T
    plot_box.columns = methods

    # Plot the results

    # Boxplot MSE
    sns.boxplot(
        data=plot_box,
        ax=axd['mse'],
        palette="pastel",
        fliersize=3
    )

    axd['mse'].set_title('MSE Distribution')
    axd['mse'].set_xlabel('Method')
    axd['mse'].set_ylabel('MSE')
    axd['mse'].tick_params(pad=-3)

    # Calculate mean and standard error for coverage
    cov_means = [np.mean(c) for c in cov_data]
    cov_se = [np.std(c) / np.sqrt(len(c)) for c in cov_data]

    # Make covdata a dataframe
    cov_df = pd.DataFrame(cov_data).T
    cov_df.columns = methods

    # Coverage errorbar (point + bar)
    sns.pointplot(
        data=cov_df,
        ax=axd['coverage'],
        palette="pastel",
        errorbar=('se',1.96),
        capsize=0,
        markersize=3,
        err_kws={'linewidth': 1}
    )

    # add horizontal line for nominal coverage
    axd['coverage'].axhline(0.95, color='gray', linestyle='--', linewidth=1, label='Nominal 95%')
    axd['coverage'].set_ylim(-0.05, 1.05)
    axd['coverage'].set_title('Empirical Coverage')
    axd['coverage'].set_ylabel('Coverage')
    axd['coverage'].set_xlabel('Method')
    axd['coverage'].tick_params(pad=-3)

    plt.savefig('sim_coef.pdf', bbox_inches='tight')