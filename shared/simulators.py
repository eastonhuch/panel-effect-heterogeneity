import numpy as np
from abc import ABC, abstractmethod
from typing import Union, List, Literal
from shared.utils import chol_inv_stack


class BaseEffectSimulator(ABC):
    name: str
    N: int
    i: int  # Index of the effect simulator in the collection

    @abstractmethod
    def sample_params(self):
        pass
    
    @abstractmethod
    def simulate_y1(self, x: np.ndarray, y0: np.ndarray) -> np.ndarray:
        pass

    def set_i(self, i: int) -> 'BaseEffectSimulator':
        self.i = i
        return self


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
        y1 = (y0.T + self.theta).T
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
        y1 = (y0.T + self.theta + x.T * self.beta).T
        if self.epsilon_sd is not None:
            y1 += np.random.normal(scale=self.epsilon_sd, size=y1.shape)
        return y1
    
class EffectSimulatorCollection():
    def __init__(self, effect_simulators: Union[None, BaseEffectSimulator, List[BaseEffectSimulator]] = None):
        self.names = []
        self.num_effect_simulators = 0
        self.effect_simulators = []
        self.N = None
        if effect_simulators is None:
            effect_simulators = []
        elif not isinstance(effect_simulators, list):
            effect_simulators = [effect_simulators]
        for effect_simulator in effect_simulators:
            self.add_effect_simulator(effect_simulator)

    def add_effect_simulator(self, effect_simulator: BaseEffectSimulator) -> 'EffectSimulatorCollection':
        if effect_simulator.name in self.names:
            raise ValueError(f"Effect Simulator '{effect_simulator.name}' already exists in collection.")
        if (self.N is not None) and (self.N != effect_simulator.N):
            raise ValueError(f"Effect Simulator has N={effect_simulator.N}; I expected {self.N}.")
        effect_simulator.set_i(self.num_effect_simulators)
        self.effect_simulators.append(effect_simulator)
        self.names.append(effect_simulator.name)
        self.num_effect_simulators += 1
        return self
    
    def __iter__(self):
        self.iter_idx = 0
        return self
    
    def __next__(self):
        if self.iter_idx >= self.num_effect_simulators:
            raise StopIteration
        effect_simulator = self.effect_simulators[self.iter_idx]
        self.iter_idx += 1
        return effect_simulator


class DataSimulator():
    weight_func_dict = {
        "ATE": lambda p: np.ones_like(p),
        "ATO": lambda p: p * (1. - p),
        "ATT": lambda p: p,
        "ATC": lambda p: 1. - p,
    }

    def __init__(self, N: int, T: int):
        self.N = N
        self.T = T
        self.D = 2  # Dimension of theta
    
    def simulate_history(self, weight_type: Literal["ATE", "ATO", "ATT", "ATC"] = "ATE") -> 'DataSimulator':
        available_weight_types = list(self.weight_func_dict.keys())
        if weight_type not in available_weight_types:
            raise ValueError(f"weight_type must be in {available_weight_types}")
        weight_func = self.weight_func_dict[weight_type]

        # Create placeholders
        x = np.zeros((self.N, self.T+1))
        p = x.copy()
        y0 = x.copy()

        # Simulate intial values
        x[:, 0] = np.random.normal(size=self.N)
        y0[:, 0] = np.random.normal(size=self.N)

        # Simulate remaining time points
        for t in range(1, self.T+1):
            x_t = 0.4*x[:, t-1] + 0.2*y0[:, t-1] + np.random.normal(size=self.N)
            p_t = 1. / (1. + np.exp(-x_t/2.))
            y0_t = 0.2*x_t + 0.3*y0[:, t-1] + np.random.normal(size=self.N)
            x[:, t] = x_t
            p[:, t] = p_t
            y0[:, t] = y0_t

        # Save data from t=1...T
        # NOTE: This discards the first time point
        self.x = x[:, 1:].copy() + 1.  # Not mean centered
        self.p = p[:, 1:].copy()
        self.not_p = 1. - self.p
        self.w = weight_func(self.p.copy())
        self.y0 = y0[:, 1:].copy() - 2.  # Not mean centered
        return self
    
    def simulate_a(self) -> 'DataSimulator':
        self.a = np.random.random(size=self.p.shape) < self.p
        self.not_a = ~self.a.copy()
        if self.y1 is None:
            raise ValueError("y1 must be simulated before simulating a.")
        self.y = self.a * self.y1 + self.not_a * self.y0

        # Compute true value of estimands
        # Theta
        self.X = np.ones((self.N, self.T, 2))
        self.X[:, :, 1] = self.x.copy()
        self.sqrt_w = np.sqrt(self.w)
        self.W_sqrt_X = (self.X.T * self.sqrt_w.T).T
        self.XtWX = np.swapaxes(self.W_sqrt_X, 1, 2) @ self.W_sqrt_X
        self.XtWX_inv = chol_inv_stack(self.XtWX)
        self.WX = (self.X.T * self.w.T).T
        tau_row_vectors = self.tau.reshape((self.N, 1, self.T))
        self.XtWtau = np.swapaxes(tau_row_vectors @ self.WX, 1, 2)
        self.theta = (self.XtWX_inv @ self.XtWtau)[:, :, 0]

        # User-level effects
        w_mean = self.w.mean(axis=1)
        self.user_effects = (self.tau * self.w).mean(axis=1) / w_mean

        return self

    def simulate_effects(self, effect_generator: BaseEffectSimulator) -> 'DataSimulator':
        self.y1 = effect_generator.simulate_y1(self.x, self.y0)
        self.tau = self.y1 - self.y0
        return self

