import numpy as np
import pandas as pd
from abc import ABC, abstractmethod


class UserResults():

    def __init__(self, user_estimates, user_cis, mean_estimate, mean_ci, alpha=0.05):
        self.user_estimates = user_estimates
        self.user_cis = user_cis
        self.mean_estimate = mean_estimate
        self.mean_ci = mean_ci
        self.alpha = alpha

    def get_user_df(self):
        return pd.DataFrame({
            'user_estimate': self.user_estimates,
            'user_lower': self.user_cis[:, 0],
            'user_upper': self.user_cis[:, 1]
        })
    
    def get_mean_series(self):
        return pd.Series({
            'mean_estimate': self.mean_estimate,
            'mean_lower': self.mean_ci[0],
            'mean_upper': self.mean_ci[1]
        })