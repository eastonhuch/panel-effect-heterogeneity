import numpy as np
import pandas as pd
from shared.simulators import DataSimulator, EffectsSimulatorCollection
from shared.analyzers import AnalyzerCollection
from shared.inference_formats import UserInferences, ThetaInferences
from shared.utils import float2str_format, print_se_parentheses


class Results():
    def __init__(self, data_simulator: DataSimulator, effect_simulators: EffectsSimulatorCollection,
                 analyzers: AnalyzerCollection, n_reps: int = 1):
        self.data_simulator = data_simulator
        self.effect_simulators = effect_simulators
        self.analyzers = analyzers
        self.n_reps = n_reps
        self.N = data_simulator.N
        self.T = data_simulator.T
        self.D = data_simulator.D
        self.alpha = analyzers.alpha
        self.num_effect_simulators = effect_simulators.num_effect_simulators
        self.num_analyzers = analyzers.num_analyzers
        self.metric_names = ["Avg Bias", "Avg Sq. Bias", "MSE", "Coverage", "CI Length"]
        self.initalize_arrays()

    def initalize_arrays(self):
        # Shorthand for dimensions
        R = self.n_reps
        E = self.num_effect_simulators
        A = self.num_analyzers

        # Theta arrays
        self.thetas = np.zeros((R, E, A, self.N, self.D))
        self.mean_thetas = np.zeros((R, E, A, self.D))
        self.theta_estimates = np.zeros((R, E, A, self.N, self.D))
        self.theta_lower_bounds = np.zeros((R, E, A, self.N, self.D))
        self.theta_upper_bounds = np.zeros((R, E, A, self.N, self.D))
        self.mean_theta_estimates = np.zeros((R, E, A, self.D))
        self.mean_theta_lower_bounds = np.zeros((R, E, A, self.D))
        self.mean_theta_upper_bounds = np.zeros((R, E, A, self.D))

        # User arrays
        self.user_effects = np.zeros((R, E, A, self.N))
        self.mean_user_effects = np.zeros((R, E, A))
        self.user_estimates = np.zeros((R, E, A, self.N))
        self.user_lower_bounds = np.zeros((R, E, A, self.N))
        self.user_upper_bounds = np.zeros((R, E, A, self.N))
        self.mean_user_estimates = np.zeros((R, E, A))
        self.mean_user_lower_bounds = np.zeros((R, E, A))
        self.mean_user_upper_bounds = np.zeros((R, E, A))

    def add_theta_inferences(
            self, theta_inferences: ThetaInferences, thetas: np.ndarray,
            rep_idx: int, effect_simulator_idx: int, analyzer_idx: int):
        r = rep_idx
        e = effect_simulator_idx
        a = analyzer_idx
        self.thetas[r, e, a] = thetas.copy()
        self.mean_thetas[r, e, a] = thetas.mean(axis=0).copy()
        self.theta_estimates[r, e, a] = theta_inferences.theta_estimates.copy()
        self.theta_lower_bounds[r, e, a] = theta_inferences.theta_lower_bounds.copy()
        self.theta_upper_bounds[r, e, a] = theta_inferences.theta_upper_bounds.copy()
        self.mean_theta_estimates[r, e, a] = theta_inferences.mean_estimate.copy()
        self.mean_theta_lower_bounds[r, e, a] = theta_inferences.mean_lower_bounds.copy()
        self.mean_theta_upper_bounds[r, e, a] = theta_inferences.mean_upper_bounds.copy()
        return self

    def add_user_inferences(
            self, user_inferences: UserInferences, user_effects: np.ndarray,
            rep_idx: int, effect_simulator_idx: int, analyzer_idx: int):
        r = rep_idx
        e = effect_simulator_idx
        a = analyzer_idx
        self.user_effects[r, e, a] = user_effects.copy()
        self.mean_user_effects[r, e, a] = user_effects.mean(axis=0).copy()
        self.user_estimates[r, e, a] = user_inferences.user_estimates.copy()
        self.user_lower_bounds[r, e, a] = user_inferences.user_lower_bounds.copy()
        self.user_upper_bounds[r, e, a] = user_inferences.user_upper_bounds.copy()
        self.mean_user_estimates[r, e, a] = user_inferences.mean_estimate.copy()
        self.mean_user_lower_bounds[r, e, a] = user_inferences.mean_lower_bounds.copy()
        self.mean_user_upper_bounds[r, e, a] = user_inferences.mean_upper_bounds.copy()
        return self
    
    def create_results_df(self, errors, covered, lengths, digits=3):
        bias = errors.mean(axis=0)
        avg_bias = float2str_format((bias).mean(axis=2), digits)
        avg_sq_bias = float2str_format((np.square(bias)).mean(axis=2), digits)

        mse_reps = (errors**2).mean(axis=3)
        mse_avg = mse_reps.mean(axis=0)
        mse_se = mse_reps.std(axis=0) / np.sqrt(self.n_reps)
        mse = print_se_parentheses(mse_avg, mse_se, digits)

        coverage_reps = covered.mean(axis=3)
        coverage_avg = coverage_reps.mean(axis=0)
        coverage_se = coverage_reps.std(axis=0) / np.sqrt(self.n_reps)
        coverage = print_se_parentheses(coverage_avg, coverage_se, digits)

        lengths_reps = lengths.mean(axis=3)
        lengths_avg = lengths_reps.mean(axis=0)
        lengths_se = lengths_reps.std(axis=0) / np.sqrt(self.n_reps)
        ci_lengths = print_se_parentheses(lengths_avg, lengths_se, digits)

        results_list = [avg_bias, avg_sq_bias, mse, coverage, ci_lengths]
        results_raw = np.stack(results_list)
        D_ = results_raw.shape[3]
        results = np.moveaxis(results_raw, [0, 1, 2, 3], [1, 2, 3, 0])
        results_2d = results.reshape(-1, results.shape[3])
        if D_ == 1:
            coef_names = ["Intercept"]
        elif D_ == 2:
            coef_names = ["Intercept", "Slope"]
        else:
            coef_names = [f"Coef. {i}" for i in range(D_)]
        result_index = pd.MultiIndex.from_product(
            [coef_names, self.metric_names, self.effect_simulators.names],
            names=["Coefficient", "Metric", "Effect Simulator"])
        df = pd.DataFrame(results_2d, index=result_index, columns=self.analyzers.names)
        return df

    def process(self, digits=3):
        # Theta (R, E, A, N, D)
        theta_errors = self.theta_estimates - self.thetas
        theta_covered = (self.theta_lower_bounds <= self.thetas) & (self.thetas <= self.theta_upper_bounds)
        theta_lengths = self.theta_upper_bounds - self.theta_lower_bounds
        self.theta_results_df = self.create_results_df(
            theta_errors, theta_covered, theta_lengths, digits)

        # Mean theta (R, E, A, D)
        mean_theta_errors = self.mean_theta_estimates - self.mean_thetas
        mean_thetas_covered = (self.mean_theta_lower_bounds <= self.mean_thetas) & (self.mean_thetas <= self.mean_theta_upper_bounds)
        mean_theta_lengths = self.mean_theta_upper_bounds - self.mean_theta_lower_bounds
        self.mean_theta_results_df = self.create_results_df(
            mean_theta_errors[:, :, :, np.newaxis, :],
            mean_thetas_covered[:, :, :, np.newaxis, :],
            mean_theta_lengths[:, :, :, np.newaxis, :],
            digits)

        # User effects (R, E, A, N)
        user_errors = self.user_estimates - self.user_effects
        user_covered = (self.user_lower_bounds <= self.user_effects) & (self.user_effects <= self.user_upper_bounds)
        user_lengths = self.user_upper_bounds - self.user_lower_bounds
        self.user_results_df = self.create_results_df(
            user_errors[:, :, :, :, np.newaxis],
            user_covered[:, :, :, :, np.newaxis],
            user_lengths[:, :, :, :, np.newaxis],
            digits)

        # Mean user effects (R, E, A)
        mean_user_errors = self.mean_user_estimates - self.mean_user_effects
        mean_user_covered = (self.mean_user_lower_bounds <= self.mean_user_effects) & (self.mean_user_effects <= self.mean_user_upper_bounds)
        mean_user_lengths = self.mean_user_upper_bounds - self.mean_user_lower_bounds
        self.mean_user_results_df = self.create_results_df(
            mean_user_errors[:, :, :, np.newaxis, np.newaxis],
            mean_user_covered[:, :, :, np.newaxis, np.newaxis],
            mean_user_lengths[:, :, :, np.newaxis, np.newaxis],
            digits)

        return self
