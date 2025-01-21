import pandas as pd


class UserInferences():
    def __init__(self, user_estimates, user_lower_bounds, user_upper_bounds,
                 mean_estimate, mean_lower_bounds, mean_upper_bounds, alpha=0.05):
        self.user_estimates = user_estimates
        self.user_lower_bounds = user_lower_bounds
        self.user_upper_bounds = user_upper_bounds
        self.mean_estimate = mean_estimate
        self.mean_lower_bounds = mean_lower_bounds
        self.mean_upper_bounds = mean_upper_bounds
        self.alpha = alpha

    def get_user_df(self):
        return pd.DataFrame({
            'Estimate': self.user_estimates,
            'Lower Bound': self.user_lower_bounds,
            'Upper Bound': self.user_upper_bounds
        })
    
    def get_mean_series(self):
        return pd.Series({
            'Estimate': self.mean_estimate,
            'Lower Bound': self.mean_lower_bounds,
            'Upper Bound': self.mean_upper_bounds
        })


class ThetaInferences():
    def __init__(self, theta_estimates, theta_lower_bounds, theta_upper_bounds,
                 mean_estimate, mean_lower_bounds, mean_upper_bounds, alpha=0.05):
        self.theta_estimates = theta_estimates
        self.theta_lower_bounds = theta_lower_bounds
        self.theta_upper_bounds = theta_upper_bounds
        self.mean_estimate = mean_estimate
        self.mean_lower_bounds = mean_lower_bounds
        self.mean_upper_bounds = mean_upper_bounds
        self.alpha = alpha
