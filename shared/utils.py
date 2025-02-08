import numpy as np

def chol_inv_matrix(x):
    chol = np.linalg.cholesky(x)
    chol_inv = np.linalg.solve(chol, np.eye(x.shape[0]))
    inv = chol_inv.T @ chol_inv
    return inv

def chol_inv_stack(x):
    chols = np.linalg.cholesky(x)
    identities = np.zeros_like(x)
    identities[:] = np.eye(x.shape[1])
    chol_invs = np.linalg.solve(chols, identities)
    invs = np.swapaxes(chol_invs, 1, 2) @ chol_invs
    return invs

def float2str_format(x: np.ndarray, digits=3) -> np.ndarray:
    fmt = f"%.{digits}f"
    x_str = np.char.mod(fmt, x)
    return x_str

def print_se_parentheses(x: np.ndarray, x_ses: np.ndarray, digits=3) -> np.ndarray:
    x_str = float2str_format(x, digits)
    x_se_str = float2str_format(x_ses, digits)
    x_se_parentheses = np.char.add(x_str, np.char.add(" (", np.char.add(x_se_str, ")")))
    return x_se_parentheses

def re_mme(theta_hats, Sigmas, min_e_value=0.1):
    # Implementation of the method of moments estimator
    # for the random effects covariance matrix from
    # "Extending DerSimonian and Laird’s methodology to
    # perform multivariate random effects meta-analyses"
    # by Jackson et al. (2008)
    # https://doi.org/10.1002/sim.3602
    # We set a positive minimum eigenvalue to ensure positive definiteness
    N, D = theta_hats.shape
    variances = np.diagonal(Sigmas, axis1=1, axis2=2)
    sds = np.sqrt(variances)
    sds_3d = sds[:, :, np.newaxis]
    weights = 1. / (sds_3d @ np.swapaxes(sds_3d, 1, 2))
    weight_means = weights.mean(axis=0)
    theta_hats_repped = np.repeat(theta_hats[:, np.newaxis, :], D, axis=1)
    theta_hat_bars = (theta_hats_repped * weights).mean(axis=0) / weight_means
    diffs = theta_hats_repped - theta_hat_bars
    outer_diffs = diffs * np.swapaxes(diffs, 1, 2)
    weighted_outer_diffs = weights * outer_diffs
    Q = weighted_outer_diffs.mean(axis=0)
    diag_es = np.diag(Q)
    inv_variances_mean = (1. / variances).mean(axis=0)
    den = inv_variances_mean - ((variances**-2).mean(axis=0) / inv_variances_mean) / N
    re_vars = (diag_es - (N-1)/N) / den
    corrs = (np.swapaxes(Sigmas.T / sds.T, 0, 1) / sds.T).T
    a = corrs.mean(axis=0) - ((corrs * weights).mean(axis=0) / weight_means) / N
    b = weight_means - ((weights**2).mean(axis=0) / weight_means) / N
    re_covariances = (Q - a) / b
    re_cov = re_covariances.copy()
    re_cov[range(D), range(D)] = re_vars
    e_values, e_vectors = np.linalg.eig(re_cov)
    e_values[e_values < min_e_value] = min_e_value
    sqrt_e_values = np.sqrt(e_values)
    re_cov_pd_left = sqrt_e_values * e_vectors
    re_cov_pd = re_cov_pd_left @ re_cov_pd_left.T
    
    re_cov_pd_Sigmas = re_cov_pd + Sigmas
    re_cov_pd_Sigmas_inv = chol_inv_stack(re_cov_pd_Sigmas)
    re_cov_pd_Sigmas_inv_sum = re_cov_pd_Sigmas_inv.sum(axis=0)
    re_cov_pd_Sigmas_inv_sum_inv = chol_inv_matrix(re_cov_pd_Sigmas_inv_sum)
    re_cov_pd_Sigmas_inv_theta_hat_sum = (re_cov_pd_Sigmas_inv @ theta_hats[:, :, np.newaxis]).sum(axis=0)
    re_mean = (re_cov_pd_Sigmas_inv_sum_inv @ re_cov_pd_Sigmas_inv_theta_hat_sum)[:, 0]
    re_mean_cov = re_cov_pd_Sigmas_inv_sum_inv

    mean = theta_hats.mean(axis=0)
    mean_cov = Sigmas.mean(axis=0) / N

    return re_mean, re_mean_cov, re_cov_pd, mean, mean_cov