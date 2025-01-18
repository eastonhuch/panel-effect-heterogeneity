import numpy as np
from abc import ABC, abstractmethod
from typing import Union


class BaseEffectSimulator(ABC):
    name: str
    N: int

    @abstractmethod
    def sample_params(self):
        pass
    
    @abstractmethod
    def simulate_y1(self, x: np.ndarray, y0: np.ndarray) -> np.ndarray:
        pass


class ConstantEffectSimulator(BaseEffectSimulator):
    def __init__(self, name: str, N: int, mu_mean: float, mu_sd: float,
                 tau: float, epsilon_sd: Union[None, float]=None):
        self.name = name
        self.N = N
        self.mu_mean = mu_mean
        self.mu_sd = mu_sd
        self.tau = tau
        self.epsilon_sd = epsilon_sd
        
    def sample_params(self) -> 'ConstantEffectSimulator':
        self.mu = np.random.normal(loc=self.mu_mean, scale=self.mu_sd)
        self.theta = np.random.normal(loc=self.mu, scale=self.tau, size=self.N)
        return self
        
    def simulate_y1(self, x: np.ndarray, y0: np.ndarray) -> np.ndarray:
        y1 = y0 + self.theta
        if self.epsilon_sd is not None:
            y1 += np.random.normal(scale=self.epsilon_sd, size=y1.shape)
        return y1
    

class ModeratedEffectSimulator(BaseEffectSimulator):
    def __init__(self, name: str, N: int, mu_mean: float, mu_sd: float,
                 tau: float, beta_mean: float, beta_sd: float, 
                 epsilon_sd: Union[None, float]=None):
        self.name = name
        self.N = N
        self.mu_mean = mu_mean
        self.mu_sd = mu_sd
        self.tau = tau
        self.beta_mean = beta_mean
        self.beta_sd = beta_sd
        self.epsilon_sd = epsilon_sd
        
    def sample_params(self) -> 'ModeratedEffectSimulator':
        self.mu = np.random.normal(loc=self.mu_mean, scale=self.mu_sd)
        self.theta = np.random.normal(loc=self.mu, scale=self.tau, size=self.N)
        self.beta = np.random.normal(loc=self.beta_mean, scale=self.beta_sd, size=self.N)
        return self
        
    def simulate_y1(self, x: np.ndarray, y0: np.ndarray) -> np.ndarray:
        y1 = y0 + self.theta + x * self.beta
        if self.epsilon_sd is not None:
            y1 += np.random.normal(scale=self.epsilon_sd, size=y1.shape)
        return y1
    

class DataSimulator():
    def __init__(self, N: int, T: int):
        self.N = N
        self.T = T

    def simulate(self, effect_generator: BaseEffectSimulator) -> 'DataSimulator':
        # Create placeholders
        x = np.zeros((self.N, self.T+1))
        p = x.copy()
        a = x.copy().astype(bool)
        y0 = x.copy()
        y1 = x.copy()
        y = x.copy()

        # Simulate intial values
        x[:, 0] = np.random.normal(size=self.N)
        y0[:, 0] = np.random.normal(size=self.N)
        y[:, 0] = y0[:, 0]

        # Simulate remaining time points
        for t in range(1, self.T+1):
            x_t = 0.4*x[:, t-1] + 0.2*y0[:, t-1] + np.random.normal(size=self.N)
            p_t = 1. / (1. + np.exp(-x_t/2.))
            a_t = np.random.random(size=p_t.shape) < p_t
            y0_t = 0.2*x_t + 0.3*y[:, t-1] + np.random.normal(size=self.N)
            y1_t = effect_generator.simulate_y1(x_t, y0_t)
            y_t = a_t*y1_t + (1 - a_t)*y0_t            
            x[:, t] = x_t
            p[:, t] = p_t
            a[:, t] = a_t
            y0[:, t] = y0_t
            y1[:, t] = y1_t
            y[:, t] = y_t

        # Save data from t=1...T
        # NOTE: This discards the first time point
        self.x = x[:, 1:]
        self.p = x[:, 1:]
        self.a = a[:, 1:]
        self.not_a = ~self.a.copy()
        self.y0 = y0[:, 1:]
        self.y1 = y1[:, 1:]
        self.y = y[:, 1:]
        self.ites = self.y1 - self.y0

        return self
