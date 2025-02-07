import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod
from typing import Union, List, Tuple
from shared.simulators import DataSimulator
from shared.inference_formats import UserInferences, ThetaInferences
from shared.utils import chol_inv_matrix, chol_inv_stack, re_mme


class BaseAnalyzer(ABC):
    name: str
    alpha: float

    def __init__(self, name: str, alpha: float=0.05):
        self.name = name
        self.alpha = alpha

    @abstractmethod
    def get_inferences(self, data_simulator: DataSimulator, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        pass
        
    def get_theta_inferences(self, data_simulator: DataSimulator) -> ThetaInferences:
        X = np.ones((data_simulator.N, data_simulator.T, 2))
        X[:, :, 1] = data_simulator.x.copy()
        inferences = self.get_inferences(data_simulator, X)
        theta_inferences = ThetaInferences(*inferences, self.alpha)
        return theta_inferences

    def get_user_inferences(self, data_simulator: DataSimulator) -> UserInferences:
        X = np.ones((data_simulator.N, data_simulator.T, 1))
        inferences = self.get_inferences(data_simulator, X)
        user_inferences = UserInferences(*[x.squeeze() for x in inferences], self.alpha)
        return user_inferences
    
    def set_i(self, i: int) -> 'BaseAnalyzer':
        self.i = i
        return self


class StandaloneAnalyzer(BaseAnalyzer):
    @abstractmethod
    def get_estimates_mean_cov(self, data_simulator: DataSimulator, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        pass

    def get_inferences(self, data_simulator: DataSimulator, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        estimates, estimates_cov = self.get_estimates_mean_cov(data_simulator, X)
        N, _ = estimates.shape
        theta_vars = np.diagonal(estimates_cov, axis1=1, axis2=2)
        theta_ses = np.sqrt(theta_vars)
        z_star = norm.ppf(1. - self.alpha / 2.)
        u = estimates
        u_lb = estimates - z_star * theta_ses
        u_ub = estimates + z_star * theta_ses
        m = u.mean(axis=0)
        m_var = theta_vars.mean(axis=0) / N
        m_se = np.sqrt(m_var)
        m_lb = m - z_star * m_se
        m_ub = m + z_star * m_se
        return u, u_lb, u_ub, m, m_lb, m_ub
    

class AnalyzerCollection():
    def __init__(self, analyzers: Union[None, BaseAnalyzer, List[BaseAnalyzer]] = None):
        self.names = []
        self.num_analyzers = 0
        self.analyzers = []
        self.alpha = None
        if analyzers is None:
            analyzers = []
        if not isinstance(analyzers, list):
            analyzers = [analyzers]
        for analyzer in analyzers:
            self.add_analyzer(analyzer)

    def add_analyzer(self, analyzer: BaseAnalyzer) -> 'AnalyzerCollection':
        if analyzer.name in self.names:
            raise ValueError(f"Analyzer name '{analyzer.name}' already exists.")
        if (self.alpha is not None) and (self.alpha != analyzer.alpha):
            raise ValueError(f"Effect Simulator has alpha={analyzer.alpha}; I expected {self.alpha}.")

        analyzer.set_i(self.num_analyzers)
        self.analyzers.append(analyzer)
        self.names.append(analyzer.name)
        self.num_analyzers += 1
        return self
    
    def __iter__(self):
        self.iter_idx = 0
        return self
    
    def __next__(self):
        if self.iter_idx >= self.num_analyzers:
            raise StopIteration
        analyzer = self.analyzers[self.iter_idx]
        self.iter_idx += 1
        return analyzer


# Begin Analyzer implementations
class WLSMixin():
    """Mixin that adds WLS estimates"""
    def wls(self, data_simulator: DataSimulator, X_raw: np.ndarray):
        # Extract data
        N, T, D = X_raw.shape
        a = data_simulator.a.copy()
        not_p = data_simulator.not_p.copy()
        y = data_simulator.y.copy()
        w = data_simulator.w.copy()
        sqrt_w = data_simulator.sqrt_w.copy()

        # Construct X
        X = np.zeros((N, T, 2*D))
        X[:, :, :D] = X_raw.copy()
        X0 = X.copy()
        X[:, :, D:] = (X_raw.copy().T * a.T).T
        X1 = X0.copy()
        X1[:, :, D:] = X_raw.copy()

        # Fit regression model
        W_sqrt_X = (X.T * sqrt_w.T).T
        XtWX = np.swapaxes(W_sqrt_X, 1, 2) @ W_sqrt_X
        XtWX_inv = chol_inv_stack(XtWX)
        WX = (X.T * w.T).T
        y_row_vectors = y[:, np.newaxis, :]
        XtWy = np.swapaxes(y_row_vectors @ WX, 1, 2)
        theta_full_3d = (XtWX_inv @ XtWy)

        # Construct fitted values, residuals
        fitted_values = (X @ theta_full_3d)[:, :, 0]
        fitted_values0 = (X0 @ theta_full_3d)[:, :, 0]
        fitted_values1 = (X1 @ theta_full_3d)[:, :, 0]
        residuals = y - fitted_values

        # Estimate covariance matrix
        W_sqrt_R_X = (X.T * (residuals.T * sqrt_w.T)).T
        meat = np.swapaxes(W_sqrt_R_X, 1, 2) @ W_sqrt_R_X
        theta_full_cov = XtWX_inv @ meat @ XtWX_inv

        # Create tau_hat_dr
        beta1_minus_beta0 = theta_full_3d[:, D:]
        estimated_effects = (X_raw @ beta1_minus_beta0)[:, :, 0]
        tau_hat_dr = residuals / (a - not_p) + estimated_effects

        # Return results
        results = {
            "theta_full_3d": theta_full_3d,
            "theta_full_cov": theta_full_cov,
            "residuals": residuals,
            "tau_hat_dr": tau_hat_dr,
            "fitted_values0": fitted_values0,
            "fitted_values1": fitted_values1}

        return results
    

class TauMixin():
    def get_tau_hat(self, data_simulator: DataSimulator, X: np.ndarray):
        y = data_simulator.y
        a = data_simulator.a
        p = data_simulator.p
        not_a = data_simulator.not_a
        not_p = data_simulator.not_p
        tau_hat = y * (a/p - not_a/not_p)
        return tau_hat
    

class WLSAnalyzer(StandaloneAnalyzer, WLSMixin):
    """Weighted least squares analyzer.
    In general, this method is biased for the treatment effects.
    We include it as a baseline for comparison.
    """
    def get_estimates_mean_cov(self, data_simulator: DataSimulator, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Get WLS estimates
        N, T, D = X.shape
        wls_results = self.wls(data_simulator, X)
        theta_full_3d = wls_results["theta_full_3d"]
        theta_full_cov = wls_results["theta_full_cov"]
        estimates = theta_full_3d[:, D:, 0]
        estimates_cov = theta_full_cov[np.ix_(range(N), range(D, 2*D), range(D, 2*D))]
        return estimates, estimates_cov


class IPWAnalyzer(StandaloneAnalyzer, WLSMixin, TauMixin):
    """Inverse-probability weighting analyzer.
    This approach is unbiased, but is not very efficient.
    """
    def __init__(self, dr: bool = False, robust: bool = True, **kwargs):
        self.dr = dr
        self.robust = robust
        super().__init__(**kwargs)

    def get_estimates_mean_cov(self, data_simulator: DataSimulator, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Extract data
        N, T, D = X.shape
        a = data_simulator.a.copy()
        p = data_simulator.p.copy()
        y = data_simulator.y.copy()
        w = data_simulator.w.copy()
        sqrt_w = data_simulator.sqrt_w.copy()

        # Get tau_hat
        if self.dr:
            wls_results = self.wls(data_simulator, X)
            residuals = wls_results["residuals"]
            tau_hat = wls_results["tau_hat_dr"]
            fitted0 = wls_results["fitted_values0"]
            fitted1 = wls_results["fitted_values1"]
        else:
            residuals = y.copy()
            tau_hat = self.get_tau_hat(data_simulator, X)
            fitted0 = np.zeros_like(y)
            fitted1 = np.zeros_like(y)
        tau_hat_row_vectors = tau_hat[:, np.newaxis, :]

        # Generate estimates
        W_sqrt_X = (X.T * sqrt_w.T).T
        XtWX = np.swapaxes(W_sqrt_X, 1, 2) @ W_sqrt_X
        XtWX_inv = chol_inv_stack(XtWX)
        WX = (X.T * data_simulator.w.T).T
        XtWtau_hat = np.swapaxes(tau_hat_row_vectors @ WX, 1, 2)
        estimates_3d = (XtWX_inv @ XtWtau_hat)
        estimated_effects = (X @ estimates_3d)[:, :, 0]
        estimates = estimates_3d[:, :, 0]

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
        W_S_X = (X.T * (s*w).T).T
        meat = np.swapaxes(W_S_X, 1, 2) @ W_S_X
        estimates_cov = XtWX_inv @ meat @ XtWX_inv

        return estimates, estimates_cov


class SIPWAnalyzer(StandaloneAnalyzer, WLSMixin, TauMixin):
    """Stabilized inverse-probability weighting analyzer.
    This approach is unbiased and more efficient than standard IPW.
    """
    def __init__(self, dr: bool = False, robust: bool = True, **kwargs):
        self.dr = dr
        self.robust = robust
        super().__init__(**kwargs)

    def get_estimates_mean_cov(self, data_simulator: DataSimulator, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # Extract data
        N, T, D = X.shape
        a = data_simulator.a.copy()
        not_a = data_simulator.not_a.copy()
        p = data_simulator.p.copy()
        not_p = data_simulator.not_p.copy()
        y = data_simulator.y.copy()
        w = data_simulator.w.copy()
        sqrt_w = data_simulator.sqrt_w.copy()

        # Get tau_hat
        if self.dr:
            wls_results = self.wls(data_simulator, X)
            residuals = wls_results["residuals"]
            tau_hat = wls_results["tau_hat_dr"]
            fitted0 = wls_results["fitted_values0"]
            fitted1 = wls_results["fitted_values1"]
        else:
            residuals = y.copy()
            tau_hat = self.get_tau_hat(data_simulator, X)
            fitted0 = np.zeros_like(y)
            fitted1 = np.zeros_like(y)

        # Get estimates
        awp0 = not_a*w/not_p
        awp1 = a*w/p
        AWP0X = (X.T * np.sqrt(awp0.T)).T
        AWP1X = (X.T * np.sqrt(awp1.T)).T
        B0 = (np.swapaxes(AWP0X, 1, 2) @ AWP0X) / T
        B0_inv = chol_inv_stack(B0)
        B1 = (np.swapaxes(AWP1X, 1, 2) @ AWP1X) / T
        B1_inv = chol_inv_stack(B1)
        z0 = (X.T * (not_a * w * tau_hat).T).T.mean(axis=1)[:, :, np.newaxis]  # N x D x 1
        z1 = (X.T * (a * w * tau_hat).T).T.mean(axis=1)[:, :, np.newaxis]  # N x D x 1
        estimates_3d = B0_inv @ z0 + B1_inv @ z1
        estimated_effects = (X @ estimates_3d)[:, :, 0]
        estimates = estimates_3d[:, :, 0]

        # Estimate covariance
        W_sqrt_X = (X.T * sqrt_w.T).T
        B = (np.swapaxes(W_sqrt_X, 1, 2) @ W_sqrt_X) / T
        B_inv = chol_inv_stack(B)
        B_inv_rep = np.repeat(B_inv[:, np.newaxis, :, :], T, axis=1)
        B_double_inv = B_inv @ B_inv

        if self.robust:
            # Control
            B_double_inv_z0 = (B_double_inv @ z0)[:, :, 0]  # N x D
            B_double_inv_z0_rep = np.repeat(B_double_inv_z0[:, np.newaxis, :], T, axis=1)  # N x T x D
            v0_right_tmp = (X * B_double_inv_z0_rep).sum(axis=2)  # N x T
            v01_left = (B_inv_rep @ ((X.T * residuals.T).T[:, :, :, np.newaxis]))[:, :, :, 0]  # N x T x D
            v0_right = (X.T * v0_right_tmp.T).T  # N x T x D
            v0 = v01_left + v0_right  # N x T x D
            v0_awp0 = (v0.T * awp0.T).T
            V0 = np.swapaxes(v0_awp0, 1, 2) @ v0_awp0

            # Treatment
            B_double_inv_z1 = (B_double_inv @ z1)[:, :, 0]  # N x D
            B_double_inv_z1_rep = np.repeat(B_double_inv_z1[:, np.newaxis, :], T, axis=1)  # N x T x D
            v1_right_tmp = (X * B_double_inv_z1_rep).sum(axis=2)  # N x T
            v1_right = (X.T * v1_right_tmp.T).T  # N x T x D
            v1 = v01_left - v1_right  # N x T x D
            v1_awp1 = (v1.T * awp1.T).T
            V1 = np.swapaxes(v1_awp1, 1, 2) @ v1_awp1

            # Combine
            estimates_cov = (V0 + V1) / T**2
        else:
            # Helper variables
            y0_imputed = y - a * estimated_effects
            y1_imputed = y + (1.-a) * estimated_effects
            r0 = y0_imputed - fitted0
            r1 = y1_imputed - fitted1

            # Control
            z0_tilde = (X.T * (w*r0).T).T.mean(axis=1)[:, :, np.newaxis]  # N x D x 1
            B_double_inv_z0_tilde = (B_double_inv @ z0_tilde)[:, :, 0]  # N x D
            B_double_inv_z0_tilde_rep = np.repeat(B_double_inv_z0_tilde[:, np.newaxis, :], T, axis=1)  # N x T x D
            v0_right_tmp = (X * B_double_inv_z0_tilde_rep).sum(axis=2)  # N x T
            v0_left = (B_inv_rep @ ((X.T * r0.T).T[:, :, :, np.newaxis]))[:, :, :, 0]  # N x T x D
            v0_right = (X.T * v0_right_tmp.T).T  # N x T x D
            v0 = v0_left - v0_right  # N x T x D

            # Treatment
            z1_tilde = (X.T * (w*r1).T).T.mean(axis=1)[:, :, np.newaxis]  # N x D x 1
            B_double_inv_z1_tilde = (B_double_inv @ z1_tilde)[:, :, 0]  # N x D
            B_double_inv_z1_tilde_rep = np.repeat(B_double_inv_z1_tilde[:, np.newaxis, :], T, axis=1)  # N x T x D
            v1_right_tmp = (X * B_double_inv_z1_tilde_rep).sum(axis=2)  # N x T
            v1_left = (B_inv_rep @ ((X.T * r1.T).T[:, :, :, np.newaxis]))[:, :, :, 0]  # N x T x D
            v1_right = (X.T * v1_right_tmp.T).T  # N x T x D
            v1 = v1_left - v1_right  # N x T x D

            # Combine
            S = (p.T*v0.T + (1.-p).T*v1.T).T
            a_var = p * (1. - p)
            sqrt_a_var = np.sqrt(a_var)
            W_Vinvsqrt_S = (S.T * (w / sqrt_a_var).T).T
            estimates_cov = (np.swapaxes(W_Vinvsqrt_S, 1, 2) @ W_Vinvsqrt_S) / (T**2)

        return estimates, estimates_cov


class MetaAnalysisMixin():
    def meta_analyze(self, raw_estimates: np.ndarray, raw_estimates_cov: np.ndarray):
        overall_mean, overall_mean_cov, overall_cov = re_mme(raw_estimates, raw_estimates_cov)
        overall_precision = chol_inv_matrix(overall_cov)
        overall_precision_mean = overall_precision @ overall_mean
        raw_estimates_precision = chol_inv_stack(raw_estimates_cov)
        estimates_precision = overall_precision + raw_estimates_precision
        estimates_cov = chol_inv_stack(estimates_precision)
        estimates_right = overall_precision_mean + raw_estimates_precision @ raw_estimates[:, :, np.newaxis]
        estimates = (estimates_cov @ estimates_right)[:, :, 0]
        return estimates, estimates_cov, overall_mean, overall_mean_cov, overall_cov


class MetaAnalyzer(BaseAnalyzer, MetaAnalysisMixin):
    """Meta-analyzer.
    This approach uses composition to obtain
    improved estimates via Gaussian meta-analysis.
    """
    def __init__(self, analyzer: StandaloneAnalyzer, **kwargs):
        self.analyzer = analyzer
        super().__init__(**kwargs)

    def get_inferences(self, data_simulator: DataSimulator, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        raw_estimates, raw_estimates_cov = self.analyzer.get_estimates_mean_cov(data_simulator, X)
        estimates, estimates_cov, overall_mean, overall_mean_cov, overall_cov = self.meta_analyze(raw_estimates, raw_estimates_cov)
        theta_vars = np.diagonal(estimates_cov, axis1=1, axis2=2)
        theta_ses = np.sqrt(theta_vars)
        z_star = norm.ppf(1. - self.alpha / 2.)
        u = estimates
        u_lb = estimates - z_star * theta_ses
        u_ub = estimates + z_star * theta_ses

        # Use the raw estimates for the mean effects
        m = overall_mean
        m_var = np.diag(overall_mean_cov)
        m_se = np.sqrt(m_var)
        m_lb = m - z_star * m_se
        m_ub = m + z_star * m_se

        return u, u_lb, u_ub, m, m_lb, m_ub