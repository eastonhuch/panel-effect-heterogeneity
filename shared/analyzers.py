import numpy as np
from scipy.stats import norm
from abc import ABC, abstractmethod
from typing import Union, List
from shared.simulators import DataSimulator
from shared.result_processors import UserResults


class BaseAnalyzer(ABC):
    name: str
    alpha: float

    def __init__(self, name: str, alpha: float=0.05):
        self.name = name,
        self.alpha = alpha

    @abstractmethod
    def fit(self, data_simulator: DataSimulator) -> 'BaseAnalyzer':
        pass
    
    @abstractmethod
    def analyze(self, data_simulator: DataSimulator, fit: bool = True) -> UserResults:
        pass
    
    def set_i(self, i: int) -> 'BaseAnalyzer':
        self.i = i
        return self


class FixedEffectsAnalyzer(BaseAnalyzer):

    def __init__(self, name: str, alpha: float=0.05):
        super().__init__(name, alpha)

    def fit(self, data_simulator: DataSimulator) -> 'FixedEffectsAnalyzer':
        a_sums = data_simulator.a.sum(axis=1)
        not_a_sums = data_simulator.not_a.sum(axis=1)
        y1_sums = (data_simulator.y * data_simulator.a).mean(axis=1)
        y0_sums = (data_simulator.y * data_simulator.not_a).mean(axis=1)
        y_sq = data_simulator.y ** 2
        y1_sq_sums = (y_sq * data_simulator.a).mean(axis=1)
        y0_sq_sums = (y_sq * data_simulator.not_a).mean(axis=1)
        y1_means = y1_sums / a_sums
        y0_means = y0_sums / not_a_sums
        y1_vars = y1_sq_sums / a_sums - y1_means ** 2
        y0_vars = y0_sq_sums / not_a_sums - y0_means ** 2
        self.user_estimates = y1_means - y0_means
        z_star = norm.cdf(1. - self.alpha / 2.)
        user_ses_sq = y1_vars / a_sums + y0_vars / not_a_sums
        user_ses = np.sqrt(user_ses_sq)
        self.user_cis = np.column_stack([
            self.user_estimates - z_star * user_ses,
            self.user_estimates + z_star * user_ses])
        self.mean_estimate = self.user_estimates.mean()
        mean_se = user_ses_sq.mean() / data_simulator.N
        self.mean_ci = np.array([
            self.mean_estimate - z_star * mean_se,
            self.mean_estimate + z_star * mean_se])
        return self

    def analyze(self, data_simulator: DataSimulator, fit: bool = True) -> UserResults:
        if fit:
            self.fit(data_simulator)
        return UserResults(self.user_estimates, self.user_cis, self.mean_estimate, self.mean_ci)
    

class AnalyzerCollection():
    def __init__(self, analyzers: Union[None, BaseAnalyzer, List[BaseAnalyzer]] = None):
        self.names = []
        self.num_analyzers = 0
        self.analyzers = []
        if analyzers is None:
            analyzers = []
        if not isinstance(analyzers, list):
            analyzers = [analyzers]
        for analyzer in analyzers:
            self.add_analyzer(analyzer)

    def add_analyzer(self, analyzer: BaseAnalyzer) -> 'AnalyzerCollection':
        if analyzer.name in self.names:
            raise ValueError(f"Analyzer name '{analyzer.name}' already exists.")
        analyzer.set_i(self.num_analyzers)
        self.analyzers.append(analyzer)
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
