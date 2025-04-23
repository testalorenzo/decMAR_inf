import numpy as np
import pickle

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import KFold
from scipy.special import expit
from scipy.stats import f
from sklearn.base import BaseEstimator, ClassifierMixin
from statsmodels.api import GLM, families
from itertools import product

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

def generate_data(n, N, p, setting='logistic'):
    X = np.random.normal(0, 1, size=(n+N, p))
    beta = np.random.normal(0, 1, size=p)
    epsilon = np.random.normal(0, 1, size=n+N)
    Y = X @ beta + epsilon

    if setting == 'MCAR':
        pi = n / (n + N) * np.ones(n + N)
    elif setting == 'logistic':
        gamma = np.random.normal(0, 1, size=p)
        logits = np.log(n / (n + N)) + X @ gamma
        pi = expit(logits)
    else:
        raise ValueError("Invalid setting.")

    R = np.random.binomial(1, pi)
    return X, Y, R, beta, pi

def estimate_projection_model(X, Y, R, model='linear', n_splits=5):
    mu_hat = np.zeros_like(Y)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        if model == 'linear':
            reg = LinearRegression().fit(X[train_idx][R[train_idx] == 1], Y[train_idx][R[train_idx] == 1])
        elif model == 'rf':
            reg = RandomForestRegressor().fit(X[train_idx][R[train_idx] == 1], Y[train_idx][R[train_idx] == 1])
        else:
            raise ValueError("Invalid model.")
        mu_hat[test_idx] = reg.predict(X[test_idx])
    return mu_hat

def estimate_propensity_score(X, R, model='logistic', n_splits=5):
    pi_hat = np.zeros(X.shape[0])
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, test_idx in kf.split(X):
        if model == 'logistic':
            clf = LogisticRegressionWithOffset().fit(X[train_idx], R[train_idx], offset=np.log(sum(R[train_idx]) / len(R[train_idx])))
            pi_hat[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        elif model == 'rf':
            clf = RandomForestClassifier().fit(X[train_idx], R[train_idx])
            pi_hat[test_idx] = clf.predict_proba(X[test_idx])[:, 1]
        elif model == 'constant':
            clf = sum(R[train_idx]) / len(R[train_idx])
            pi_hat[test_idx] = clf
        else:
            raise ValueError("Invalid model.")
    pi_hat = np.clip(pi_hat, 1e-6, 1)  # avoid division by zero
    return pi_hat

def compute_covariance(A):
    n_total = A.shape[1]
    cov_matrix = A @ A.T / n_total
    return cov_matrix

def aipw_estimator(X, Y, R, mu_hat, pi_hat):
    n_total = X.shape[0]
    residual = (Y - mu_hat)[:,None] * R[:, None] / pi_hat[:, None]
    aipw = mu_hat + residual.reshape(n_total)
    multiplier = np.linalg.inv(X.T @ X) @ X.T
    before_sum = (multiplier * aipw) * n_total
    theta_hat = before_sum.mean(axis=1)
    cov_matrix = compute_covariance(before_sum)
    se = np.sqrt(np.diag(cov_matrix) / n_total)
    return theta_hat, se, cov_matrix

def or_estimator(X, mu_hat):
    n_total = X.shape[0]
    or_est = mu_hat
    multiplier = np.linalg.inv(X.T @ X) @ X.T
    before_sum = (multiplier * or_est) * n_total
    theta_hat = before_sum.mean(axis=1)
    cov_matrix = compute_covariance(before_sum)
    se = np.sqrt(np.diag(cov_matrix) / n_total)
    return theta_hat, se, cov_matrix

def ipw_estimator(X, Y, R, pi_hat):
    n_total = X.shape[0]
    ipw = R[:, None] / pi_hat[:, None] * Y[:, None]
    multiplier = np.linalg.inv(X.T @ X) @ X.T
    before_sum = (multiplier * ipw.reshape(n_total)) * n_total
    theta_hat = before_sum.mean(axis=1)
    cov_matrix = compute_covariance(before_sum)
    se = np.sqrt(np.diag(cov_matrix) / n_total)
    return theta_hat, se, cov_matrix

def naive_estimator(X, Y, R):
    n_total = X.shape[0]
    sub_X = X[R == 1]
    naive = Y[R == 1]
    multiplier = np.linalg.inv(sub_X.T @ sub_X) @ sub_X.T
    before_sum = (multiplier * naive) * n_total
    theta_hat = before_sum.mean(axis=1)
    cov_matrix = compute_covariance(before_sum)
    se = np.sqrt(np.diag(cov_matrix) / n_total)
    return theta_hat, se, cov_matrix

def oracle_estimator(X, Y, R, beta, pi):
    # n_total = X.shape[0]
    # theta_hat = beta.flatten()
    # cov_matrix = np.cov(Y - X @ beta, rowvar=False)
    # se = np.sqrt(np.diag(cov_matrix) / n_total)
    return [0], [0], [0]

def joint_coverage(theta_hat, cov_matrix, true_theta, n_total, alpha=0.05):
    """
    Implement Hotelling's T2 test for joint coverage.
    """
    diff = theta_hat - true_theta
    q = len(diff)
    T2 = n_total * diff.T @ np.linalg.inv(cov_matrix) @ diff
    F_stat = (n_total - q) / (q * (n_total - 1)) * T2
    p_value = 1 - f.cdf(F_stat, dfn=q, dfd=n_total - q)
    return p_value > alpha

def run_simulation(n=1000, N=10000, p=10, setting='logistic',
                   mu_model='linear', pi_model='logistic', seed=42):
    np.random.seed(seed)
    X, Y, R, beta, gamma = generate_data(n, N, p, setting)
    mu_hat = estimate_projection_model(X, Y, R, mu_model)
    pi_hat = estimate_propensity_score(X, R, pi_model)
    aipw_hat, aipw_se, aipw_cov = aipw_estimator(X, Y, R, mu_hat, pi_hat)
    or_hat, or_se, or_cov = or_estimator(X, mu_hat)
    ipw_hat, ipw_se, ipw_cov = ipw_estimator(X, Y, R, pi_hat)
    naive_hat, naive_se, naive_cov = naive_estimator(X, Y, R)
    oracle_hat, oracle_se, oracle_cov = oracle_estimator(X, Y, R, beta, gamma)
    true_theta = beta.flatten()
    return (aipw_hat, aipw_se, aipw_cov, or_hat, or_se, or_cov,
            ipw_hat, ipw_se, ipw_cov, naive_hat, naive_se, naive_cov,
            oracle_hat, oracle_se, oracle_cov, true_theta)

if __name__ == '__main__':

    n_seeds = 100
    p = 10

    ns = [100, 1000, 10000]
    N_multipliers = [10, 100]
    settings = ['logistic', 'MCAR']
    mu_models = ['rf']
    pi_models = ['logistic', 'constant']

    ext_loop = []
    combinations = product(ns, N_multipliers, settings, mu_models, pi_models)
    for n, N_multiplier, setting, mu_model, pi_model in combinations:
        N = N_multiplier * n
        print(f"Running simulation with n={n}, N={N}, setting={setting}, mu_model={mu_model}, pi_model={pi_model}")
        
        estimates = {
            'aipw': [], 'or': [], 'ipw': [], 'naive': [], 'oracle': [],
            'aipw_se': [], 'or_se': [], 'ipw_se': [], 'naive_se': [], 'oracle_se': [],
            'true_theta': [], 'aipw_cov': [], 'or_cov': [], 'ipw_cov': [], 'naive_cov': [], 'oracle_cov': []
        }
        for seed in range(n_seeds):
            results = run_simulation(n, N, p, setting, mu_model, pi_model, seed)
            (aipw_hat, aipw_se, aipw_cov, or_hat, or_se, or_cov,
             ipw_hat, ipw_se, ipw_cov, naive_hat, naive_se, naive_cov,
             oracle_hat, oracle_se, oracle_cov, true_theta) = results

            estimates['aipw'].append(aipw_hat)
            estimates['or'].append(or_hat)
            estimates['ipw'].append(ipw_hat)
            estimates['naive'].append(naive_hat)
            estimates['oracle'].append(oracle_hat)

            estimates['aipw_se'].append(aipw_se)
            estimates['or_se'].append(or_se)
            estimates['ipw_se'].append(ipw_se)
            estimates['naive_se'].append(naive_se)
            estimates['oracle_se'].append(oracle_se)

            estimates['true_theta'].append(true_theta)
            estimates['aipw_cov'].append(joint_coverage(aipw_hat, aipw_cov, true_theta, n+N))
            estimates['or_cov'].append(joint_coverage(or_hat, or_cov, true_theta, n+N))
            estimates['ipw_cov'].append(joint_coverage(ipw_hat, ipw_cov, true_theta,  n+N))
            estimates['naive_cov'].append(joint_coverage(naive_hat, naive_cov, true_theta,  n+N))
            # estimates['oracle_cov'].append(joint_coverage(oracle_hat, oracle_cov, true_theta,  n+N))

        # Store the results
        ext_loop.append({
            'n': n, 'N': N, 'setting': setting,
            'mu_model': mu_model, 'pi_model': pi_model,
            'estimates': estimates
        })

    # Save the results
    with open('simulation_results.pkl', 'wb') as f:
        pickle.dump(ext_loop, f)