import numpy as np
import pandas as pd
from scipy.stats import norm
from abc import ABC, abstractmethod
from typing import Union, List, Tuple, Optional
from shared.utils import chol_inv_matrix, chol_inv_stack, re_mme
from shared.analyzers import MetaAnalysisMixin
import copy


class BaseRaggedAnalyzer(ABC):
    name: str
    alpha: float

    def __init__(
            self, name: str = "", alpha: float = 0.05, 
            id_col: str = "id", treatment_col: str = "a", 
            prob_col: str = "p", outcome_col: str = "y",
            weight_col: Optional[str] = None):
        self.name = name
        self.alpha = alpha
        self.id_col = id_col
        self.treatment_col = treatment_col
        self.prob_col = prob_col
        self.outcome_col = outcome_col
        self.weight_col = weight_col

    def get_lists(self, df: pd.DataFrame, X: np.ndarray):
        unique_ids = df[self.id_col].unique()
        X_list = []
        a_list = []
        p_list = []
        y_list = []
        w_list = []
        for id in unique_ids:
            is_id = df[self.id_col] == id
            idx_bool = is_id.values
            idx = is_id.index
            X_list.append(X[idx_bool])
            a = df.loc[idx, self.treatment_col].values
            a_list.append(a)
            p_list.append(df.loc[idx, self.prob_col].values)
            y_list.append(df.loc[idx, self.outcome_col].values)
            if self.weight_col is not None:
                w_list.append(df[self.weight_col][idx].values)
            else:
                w_list.append(np.ones_like(a))
        return X_list, a_list, p_list, y_list, w_list

    @abstractmethod
    def get_inferences(self, df: pd.DataFrame, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Returns estimated means and covariance matrices for each user"""
        pass


class StandaloneRaggedAnalyzer(BaseRaggedAnalyzer):
    @abstractmethod
    def get_estimates_mean_cov(
            self, X_list: List[np.ndarray], a_list: List[np.ndarray], p_list: List[np.ndarray],
            y_list: List[np.ndarray], w_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_inferences(self, df: pd.DataFrame, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lists = self.get_lists(df, X)
        estimates, estimates_cov = self.get_estimates_mean_cov(*lists)
        overall_mean = estimates.mean(axis=0)
        overall_mean_cov = estimates_cov.mean(axis=0) / len(estimates)
        return estimates, estimates_cov, overall_mean, overall_mean_cov


class RaggedWLSMixin():
    """Mixin that adds WLS estimates"""
    def wls(self, X_list: List[np.ndarray], a_list: List[np.ndarray], 
            p_list: List[np.ndarray], y_list: List[np.ndarray],
            w_list: List[np.ndarray]):

        # Extract data
        N = len(X_list)
        D = X_list[0].shape[1]
        theta_full = np.zeros((N, 2*D))
        theta_full_cov = np.zeros((N, 2*D, 2*D))
        theta = np.zeros((N, D))
        theta_cov = np.zeros((N, D, D))
        residual_list = []
        tau_hat_dr_list = []
        fitted_values0_list = []
        fitted_values1_list = []

        for i in range(N):
            a = a_list[i].copy()
            p = p_list[i].copy()
            not_p = 1. - p
            y = y_list[i].copy()
            w = w_list[i].copy()
            sqrt_w = np.sqrt(w)
            X_raw = X_list[i].copy()
            T = X.shape[0]

            # Construct X
            X = np.zeros((T, 2*D))
            X[:, :D] = X_raw.copy()
            X0 = X.copy()
            X[:, D:] = (X_raw.copy().T * a).T
            X1 = X0.copy()
            X1[:, D:] = X_raw.copy()

            # Fit regression model
            W_sqrt_X = (X.T * sqrt_w).T
            XtWX = W_sqrt_X.T @ W_sqrt_X
            XtWX_inv = chol_inv_stack(XtWX)
            WX = (X.T * w).T
            XtWy = WX.T @ y
            theta_full_i = (XtWX_inv @ XtWy)[:, 1]
            theta_full[i] = theta_full_i
            theta_i = theta_full_i[D:]
            theta[i] = theta_i

            # Construct fitted values, residuals
            fitted_values = X @ theta_i
            fitted_values0_list.append(X0 @ theta_i)
            fitted_values1_list.append(X1 @ theta_i)
            residuals = y - fitted_values
            residual_list.append(residuals)

            # Estimate covariance matrix
            W_sqrt_R_X = (X.T * (residuals * sqrt_w)).T
            meat = W_sqrt_R_X.T @ W_sqrt_R_X
            theta_full_cov_i = XtWX_inv @ meat @ XtWX_inv
            theta_full_cov[i] = theta_full_cov_i
            theta_cov[i] = theta_full_cov_i[D:, D:]

            # Create tau_hat_dr
            estimated_effects = X_raw @ theta_i
            tau_hat_dr_list.append(residuals / (a - not_p) + estimated_effects)

        # Return results
        results = {
            "theta_full": theta_full,
            "theta_full_cov": theta_full_cov,
            "theta": theta,
            "theta_cov": theta_cov,
            "residual_list": residual_list,
            "tau_hat_dr_list": tau_hat_dr_list,
            "fitted_values0_list": fitted_values0_list,
            "fitted_values1_list": fitted_values1_list}

        return results
    

class RaggedTauMixin():
    def get_tau_hat_list(
            self, a_list: List[np.ndarray], 
            p_list: List[np.ndarray], y_list: List[np.ndarray]):
        tau_hat_list = []
        for i in range(len(a_list)):
            tau_hat = y_list[i] * (a_list[i] / p_list[i] - (1. - a_list[i]) / (1. - p_list[i]))
            tau_hat_list.append(tau_hat)
        return tau_hat_list
    

class WLSAnalyzer(StandaloneRaggedAnalyzer, RaggedWLSMixin):
    """Weighted least squares analyzer.
    In general, this method is biased for the treatment effects.
    We include it as a baseline for comparison.
    """
    def get_estimates_mean_cov(
            self, X_list: List[np.ndarray], a_list: List[np.ndarray], p_list: List[np.ndarray],
            y_list: List[np.ndarray], w_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        wls_results = self.wls(X_list, a_list, p_list, y_list, w_list)
        return wls_results["theta_list"], wls_results["theta_cov_list"]


class IPWAnalyzer(StandaloneRaggedAnalyzer, RaggedWLSMixin, RaggedTauMixin):
    """Inverse-probability weighting analyzer.
    This approach is unbiased, but is not very efficient.
    """
    def __init__(self, dr: bool = False, robust: bool = True, **kwargs):
        self.dr = dr
        self.robust = robust
        super().__init__(**kwargs)

    def get_estimates_mean_cov(
            self, X_list: List[np.ndarray], a_list: List[np.ndarray], p_list: List[np.ndarray],
            y_list: List[np.ndarray], w_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        N = len(X_list)
        D = X_list[0].shape[1]
        estimates = np.zeros((N, D))
        estimates_cov = np.zeros((N, D, D))

        # Get tau_hat, fitted values, and residuals
        if self.dr:
            wls_results = self.wls(X_list, a_list, p_list, y_list, w_list)
            residual_list = wls_results["residual_list"]
            tau_hat_list = wls_results["tau_hat_dr_list"]
            fitted0_list = wls_results["fitted_values0_list"]
            fitted1_list = wls_results["fitted_values1_list"]
        else:
            residual_list = copy.deepcopy(y_list)
            tau_hat_list = self.get_tau_hat(a_list, p_list, y_list)
            fitted0_list = [np.zeros_like(y) for y in y_list]
            fitted1_list = copy.deepcopy(fitted0)

        # Loop over users
        for i in range(N):
            a = a_list[i].copy()
            p = p_list[i].copy()
            not_p = 1. - p
            y = y_list[i].copy()
            w = w_list[i].copy()
            sqrt_w = np.sqrt(w)
            X = X_list[i].copy()
            T = X.shape[0]
            residuals = residual_list[i]
            tau_hat = tau_hat_list[i]
            fitted0 = fitted0_list[i]
            fitted1 = fitted1_list[i]

            # Generate estimates
            W_sqrt_X = (X.T * sqrt_w).T
            XtWX = W_sqrt_X.T @ W_sqrt_X
            XtWX_inv = chol_inv_matrix(XtWX)
            WX = (X.T * w).T
            XtWtau_hat = WX.T @ tau_hat
            estimates_i = XtWX_inv @ XtWtau_hat
            estimated_effects = X @ estimates_i
            estimates[i] = estimates_i

            # Estimate covariance
            a_var = p * (1. - p)
            if self.robust:  # Conservative Neyman estimator
                s = (a-p) * residuals / a_var
            else:  # Model-based estimates
                y0_imputed = y - a * estimated_effects
                y1_imputed = y + (1.-a) * estimated_effects
                s0 = y0_imputed - fitted0
                s1 = y1_imputed - fitted1
                s = ((1.-p) * s1 + p * s0) / np.sqrt(a_var)
            W_S_X = (X.T * (s*w)).T
            meat = W_S_X.T @ W_S_X
            estimates_cov_i = XtWX_inv @ meat @ XtWX_inv
            estimates_cov[i] = estimates_cov_i

        return estimates, estimates_cov


class SIPWAnalyzer(StandaloneRaggedAnalyzer, RaggedWLSMixin, RaggedTauMixin):
    """Stabilized inverse-probability weighting analyzer.
    This approach is unbiased and more efficient than standard IPW.
    """
    def __init__(self, dr: bool = False, robust: bool = True, **kwargs):
        self.dr = dr
        self.robust = robust
        super().__init__(**kwargs)

    def get_estimates_mean_cov(
            self, X_list: List[np.ndarray], a_list: List[np.ndarray], p_list: List[np.ndarray],
            y_list: List[np.ndarray], w_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        N = len(X_list)
        D = X_list[0].shape[1]
        estimates = np.zeros((N, D))
        estimates_cov = np.zeros((N, D, D))

        # Get tau_hat, fitted values, and residuals
        if self.dr:
            wls_results = self.wls(X_list, a_list, p_list, y_list, w_list)
            residual_list = wls_results["residual_list"]
            tau_hat_list = wls_results["tau_hat_dr_list"]
            fitted0_list = wls_results["fitted_values0_list"]
            fitted1_list = wls_results["fitted_values1_list"]
        else:
            residual_list = copy.deepcopy(y_list)
            tau_hat_list = self.get_tau_hat(a_list, p_list, y_list)
            fitted0_list = [np.zeros_like(y) for y in y_list]
            fitted1_list = copy.deepcopy(fitted0)

        # Loop over users
        for i in range(N):
            a = a_list[i].copy()
            not_a = ~a
            p = p_list[i].copy()
            not_p = 1. - p
            y = y_list[i].copy()
            w = w_list[i].copy()
            sqrt_w = np.sqrt(w)
            X = X_list[i].copy()
            T = X.shape[0]
            residuals = residual_list[i]
            tau_hat = tau_hat_list[i]
            fitted0 = fitted0_list[i]
            fitted1 = fitted1_list[i]

            # Get estimates
            awp0 = not_a*w/not_p
            awp1 = a*w/p
            AWP0X = (X.T * np.sqrt(awp0)).T
            AWP1X = (X.T * np.sqrt(awp1)).T
            B0 = (AWP0X.T @ AWP0X) / T
            B0_inv = chol_inv_matrix(B0)
            B1 = (AWP1X.T @ AWP1X) / T
            B1_inv = chol_inv_matrix(B1)
            z0 = (X.T * (not_a * w * tau_hat)).mean(axis=0)
            z1 = (X.T * (a * w * tau_hat)).mean(axis=0)
            estimates_i = B0_inv @ z0 + B1_inv @ z1
            estimated_effects = (X @ estimates_i)[:, :, 0]
            estimates[i] = estimates_i

            # Estimate covariance
            W_sqrt_X = (X.T * sqrt_w).T
            B = (W_sqrt_X.T @ W_sqrt_X) / T
            B_inv = chol_inv_matrix(B)
            B_double_inv = B_inv @ B_inv

            if self.robust:
                # Control
                v0_right_tmp = X @ (B_double_inv @ z0)
                v01_left = B_inv @ (X.T * residuals)
                v0_right = (X.T * v0_right_tmp).T
                v0 = v01_left + v0_right
                v0_awp0 = (v0.T * awp0).T
                V0 = v0_awp0.T @ v0_awp0

                # Treatment
                v1_right_tmp = X @ (B_double_inv @ z1)
                v1_right = (X.T * v1_right_tmp).T
                v1 = v01_left + v1_right
                v1_awp1 = (v1.T * awp1).T
                V1 = v1_awp1.T @ v1_awp1

                # Combine
                estimates_cov = (V0 + V1) / T**2
            else:
                # Helper variables
                y0_imputed = y - a * estimated_effects
                y1_imputed = y + not_a * estimated_effects
                r0 = y0_imputed - fitted0
                r1 = y1_imputed - fitted1

                # Control
                z0_tilde = X.T @ (w*r0)
                B_double_inv_z0_tilde = B_double_inv @ z0_tilde
                v0_right_tmp = X @ B_double_inv_z0_tilde
                v0_left = B_inv @ (X.T * r0).T
                v0_right = (X.T * v0_right_tmp).T
                v0 = v0_left - v0_right

                # Treatment
                z1_tilde = X.T @ (w*r1)
                B_double_inv_z1_tilde = B_double_inv @ z1_tilde
                v1_right_tmp = X @ B_double_inv_z1_tilde
                v1_left = B_inv @ (X.T * r1).T
                v1_right = (X.T * v1_right_tmp).T
                v1 = v1_left - v1_right

                # Combine
                S = (p*v0.T + not_p*v1.T).T
                a_var = p * not_p
                sqrt_a_var = np.sqrt(a_var)
                W_Vinvsqrt_S = (S.T * (w / sqrt_a_var)).T
                estimates_cov = (W_Vinvsqrt_S.T @ W_Vinvsqrt_S) / (T**2)

        return estimates, estimates_cov


class MetaAnalyzer(BaseRaggedAnalyzer, MetaAnalysisMixin):
    """Meta-analyzer.
    This approach uses composition to obtain
    improved estimates via Gaussian meta-analysis.
    """
    def __init__(self, analyzer: StandaloneRaggedAnalyzer, name: str, alpha: float=0.05):
        self.analyzer = analyzer
        self.name = name
        self.alpha = alpha

    def get_inferences(self, df: pd.DataFrame, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        lists = self.get_lists(df, X)
        raw_estimates, raw_estimates_cov = self.analyzer.get_estimates_mean_cov(*lists)
        estimates, estimates_cov, overall_mean, overall_mean_cov, overall_cov = self.meta_analyze(raw_estimates, raw_estimates_cov)
        return estimates, estimates_cov, overall_mean, overall_mean_cov