import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod
from typing import Union, List
from shared.simulators import DataSimulator
from shared.inference_formats import UserInferences, ThetaInferences
from shared.utils import chol_inv_stack


class BaseAnalyzer(ABC):
    name: str
    alpha: float

    def __init__(self, name: str, alpha: float=0.05):
        self.name = name
        self.alpha = alpha
    
    @abstractmethod
    def get_theta_inferences(self, data_simulator: DataSimulator) -> ThetaInferences:
        pass

    @abstractmethod
    def get_user_inferences(self, data_simulator: DataSimulator) -> UserInferences:
        pass
    
    def set_i(self, i: int) -> 'BaseAnalyzer':
        self.i = i
        return self
    
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
class WLSAnalyzer(BaseAnalyzer):
    """Weighted least squares analyzer.
    In general, this method is biased for the treatment effects.
    We include it as a baseline for comparison.
    """
    def __init__(self, name: str, alpha: float=0.05):
        super().__init__(name, alpha)

    def get_inferences(self, data_simulator: DataSimulator, X_raw: np.ndarray):
        N, T, D = X_raw.shape
        X = np.ones((N, T, 2*D))
        X[:, :, :D] = X_raw.copy()
        X[:, :, D:] = (X_raw.copy().T * data_simulator.a.T).T
        W_sqrt_X = (X.T * data_simulator.sqrt_w.T).T
        XtWX = np.swapaxes(W_sqrt_X, 1, 2) @ W_sqrt_X
        XtWX_inv = chol_inv_stack(XtWX)
        WX = (X.T * data_simulator.w.T).T
        y_row_vectors = data_simulator.y[:, np.newaxis, :]
        XtWy = np.swapaxes(y_row_vectors @ WX, 1, 2)
        theta_full_3d = (XtWX_inv @ XtWy)
        fitted_values = (X @ theta_full_3d).squeeze()
        residuals = data_simulator.y - fitted_values
        W_sqrt_R_X = (X.T * (residuals.T * data_simulator.sqrt_w.T)).T
        meat = np.swapaxes(W_sqrt_R_X, 1, 2) @ W_sqrt_R_X
        theta_full_cov = XtWX_inv @ meat @ XtWX_inv
        theta_estimates_3d = theta_full_3d[:, D:]
        u = theta_estimates_3d[:, :, 0]
        theta_cov = theta_full_cov[np.ix_(range(N), range(D, 2*D), range(D, 2*D))]
        theta_vars = np.diagonal(theta_cov, axis1=1, axis2=2)
        theta_ses = np.sqrt(theta_vars)
        z_star = norm.ppf(1. - self.alpha / 2.)
        u_lb = u - z_star * theta_ses
        u_ub = u + z_star * theta_ses
        m = u.mean(axis=0)
        m_var = theta_vars.mean(axis=0) / N
        m_se = np.sqrt(m_var)
        m_lb = m - z_star * m_se
        m_ub = m + z_star * m_se
        return u, u_lb, u_ub, m, m_lb, m_ub

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

class IPWAnalyzer(BaseAnalyzer):
    """Inverse-probability weighting analyzer.
    This approach is unbiased, but is not very efficient.
    """
    def __init__(self, name: str, alpha: float=0.05):
        super().__init__(name, alpha)

    def get_inferences(self, data_simulator: DataSimulator, X: np.ndarray):
        N, T, D = X.shape
        a = data_simulator.a.copy()
        p = data_simulator.p.copy()
        y = data_simulator.y.copy()
        sqrt_w = data_simulator.sqrt_w
        W_sqrt_X = (X.T * sqrt_w.T).T
        XtWX = np.swapaxes(W_sqrt_X, 1, 2) @ W_sqrt_X
        XtWX_inv = chol_inv_stack(XtWX)
        WX = (X.T * data_simulator.w.T).T
        tau_hat = y * (a/p - (1.-a)/(1.-p))
        tau_hat_row_vectors = tau_hat[:, np.newaxis, :]
        XtWtau_hat = np.swapaxes(tau_hat_row_vectors @ WX, 1, 2)
        theta_3d = (XtWX_inv @ XtWtau_hat)
        u = theta_3d[:, :, 0]
        s = (a-p) * y / (p * (1. - p))
        W_sqrt_S_X = (X.T * (s.T * sqrt_w.T)).T
        meat = np.swapaxes(W_sqrt_S_X, 1, 2) @ W_sqrt_S_X
        theta_cov = XtWX_inv @ meat @ XtWX_inv
        theta_vars = np.diagonal(theta_cov, axis1=1, axis2=2)
        theta_ses = np.sqrt(theta_vars)
        z_star = norm.ppf(1. - self.alpha / 2.)
        u_lb = u - z_star * theta_ses
        u_ub = u + z_star * theta_ses
        m = u.mean(axis=0)
        m_var = theta_vars.mean(axis=0) / N
        m_se = np.sqrt(m_var)
        m_lb = m - z_star * m_se
        m_ub = m + z_star * m_se
        return u, u_lb, u_ub, m, m_lb, m_ub

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
