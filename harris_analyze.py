import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from functools import partial


@dataclass
class ParameterConfig:
    """Configuration for parameter grouping and analysis"""
    name: str
    columns: List[str]
    formatter: Optional[callable] = None


class HarrisAnalyzer:
    """Unified Harris corner detector parameter analysis"""

    # Parameter mappings for readability
    APERTURE_MAP = {0: 'sobel', 1: 'prewitt', 2: 'scharr'}
    WINDOW_MAP = {0: 'uniform', 1: 'gaussian'}

    # Metric configurations
    METRICS = {
        'repeatability': {'transform': lambda x: x, 'higher_better': True},
        'localization': {'transform': lambda x: 1/(1+x), 'higher_better': True},
        'response_ratio': {'transform': lambda x: np.minimum(x, 1), 'higher_better': True}
    }

    def __init__(self, df: pd.DataFrame, distortion_no: str):
        self.df = df
        self.distortion_no = distortion_no

    def analyze_parameters(self, mode: str = 'isolated', top_n: int = 3) -> pd.DataFrame:
        """
        Unified parameter analysis with configurable modes

        Args:
            mode: 'isolated' for individual parameter analysis, 'combined' for all combinations
            top_n: Number of top results to display
        """
        if mode == 'isolated':
            return self._analyze_isolated_parameters(top_n)
        elif mode == 'combined':
            return self._analyze_combined_parameters(top_n)
        else:
            raise ValueError("Mode must be 'isolated' or 'combined'")

    def _analyze_isolated_parameters(self, top_n: int) -> pd.DataFrame:
        """Analyze individual parameters in isolation"""
        param_configs = [
            ParameterConfig('k_val', ['k_val']),
            ParameterConfig('aperture', ['aperture_tp', 'aperture_sz'],
                            self._format_aperture),
            ParameterConfig('window', ['window_tp', 'window_sz'],
                            self._format_window)
        ]

        results = []
        for config in param_configs:
            param_results = self._process_parameter_group(config)
            results.extend(param_results)

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            self._print_best_isolated(results_df)
            self._save_results(results_df, 'harris_params_isolated.parquet')

        return results_df

    def _analyze_combined_parameters(self, top_n: int) -> pd.DataFrame:
        """Analyze all parameter combinations"""
        param_columns = ['k_val', 'aperture_tp', 'aperture_sz', 'window_tp', 'window_sz']
        combinations = self.df[param_columns].drop_duplicates()

        results = []
        for _, params in combinations.iterrows():
            result = self._calculate_combination_metrics(params, param_columns)
            if result:
                results.append(result)

        results_df = pd.DataFrame(results)
        if not results_df.empty:
            self._print_top_combined(results_df, top_n)
            self._save_results(results_df, 'harris_param_combined.parquet')

        return results_df

    def _process_parameter_group(self, config: ParameterConfig) -> List[Dict]:
        """Process a single parameter group configuration"""
        unique_values = self.df[config.columns].drop_duplicates()
        results = []

        for _, value_row in unique_values.iterrows():
            # Filter data for this parameter combination
            mask = pd.Series(True, index=self.df.index)
            for col in config.columns:
                mask &= (self.df[col] == value_row[col])

            filtered_data = self.df[mask]
            if filtered_data.empty:
                continue

            # Format parameter value
            param_value = (config.formatter(value_row) if config.formatter
                           else str(value_row[config.columns[0]]))

            # Calculate metrics
            result = {'param_name': config.name, 'param_value': param_value}
            self._calculate_metrics(filtered_data, result)
            results.append(result)

        return results

    def _calculate_combination_metrics(self, params: pd.Series,
                                       param_columns: List[str]) -> Optional[Dict]:
        """Calculate metrics for a specific parameter combination"""
        # Filter data
        mask = pd.Series(True, index=self.df.index)
        for col in param_columns:
            mask &= (self.df[col] == params[col])

        param_data = self.df[mask]
        if param_data.empty:
            return None

        # Build result dictionary
        result = {
            'k_val': params['k_val'],
            'aperture_tp': self.APERTURE_MAP.get(params['aperture_tp'], str(params['aperture_tp'])),
            'aperture_sz': params['aperture_sz'],
            'window_tp': self.WINDOW_MAP.get(params['window_tp'], str(params['window_tp'])),
            'window_sz': params['window_sz'],
            'param_str': self._format_param_string(params)
        }

        self._calculate_metrics(param_data, result)
        return result

    def _calculate_metrics(self, data: pd.DataFrame, result: Dict) -> None:
        """Calculate all metrics for filtered data"""
        # Initialize metric collectors
        metric_values = {metric: [] for metric in self.METRICS}

        # Process each distortion level
        for level in range(1, 6):
            level_data = data[data['dt_lv'] == level]

            if level_data.empty:
                self._add_nan_metrics(result, level)
                continue

            # Calculate level metrics
            self._calculate_level_metrics(level_data, result, level, metric_values)

        # Calculate AUC scores
        self._calculate_auc_scores(metric_values, result)

    def _calculate_level_metrics(self, level_data: pd.DataFrame, result: Dict,
                                 level: int, metric_values: Dict) -> None:
        """Calculate metrics for a single distortion level"""
        # Repeatability
        mean_rep = level_data['mean_repeatability'].mean()
        result[f'repeatability_mean_{level}'] = mean_rep
        metric_values['repeatability'].append(mean_rep)

        # Localization
        all_loc_distances = self._collect_array_data(level_data, 'loc_distances')
        if all_loc_distances:
            mean_loc = np.mean(all_loc_distances)
            result[f'localization_mean_{level}'] = mean_loc
            metric_values['localization'].append(mean_loc)

            # Percentiles
            for pct in [50, 95, 99]:
                result[f'loc_p{pct}_{level}'] = np.percentile(all_loc_distances, pct)
        else:
            result[f'localization_mean_{level}'] = np.nan
            for pct in [50, 95, 99]:
                result[f'loc_p{pct}_{level}'] = np.nan

        # Response ratios
        all_resp_ratios = self._collect_array_data(level_data, 'resp_ratios')
        if all_resp_ratios:
            # Apply capping transformation
            capped_ratios = self.METRICS['response_ratio']['transform'](all_resp_ratios)
            mean_resp = np.mean(capped_ratios)
            result[f'response_ratio_mean_{level}'] = mean_resp
            metric_values['response_ratio'].append(mean_resp)

            # Percentiles
            for pct in [50, 95, 99]:
                result[f'resp_p{pct}_{level}'] = np.percentile(capped_ratios, pct)
        else:
            result[f'response_ratio_mean_{level}'] = np.nan
            for pct in [50, 95, 99]:
                result[f'resp_p{pct}_{level}'] = np.nan

    def _calculate_auc_scores(self, metric_values: Dict, result: Dict) -> None:
        """Calculate normalized AUC scores for all metrics"""
        for metric_name, values in metric_values.items():
            if not values:
                result[f'{metric_name}_auc'] = np.nan
                continue

            # Filter valid values and create coordinate arrays
            valid_data = [(i+1, v) for i, v in enumerate(values) if not np.isnan(v)]
            if not valid_data:
                result[f'{metric_name}_auc'] = np.nan
                continue

            x_coords, y_values = zip(*valid_data)

            # Apply metric-specific transformation
            transform = self.METRICS[metric_name]['transform']
            if metric_name == 'localization':
                y_values = [transform(v) for v in y_values]

            # Calculate normalized AUC
            if len(y_values) == 1:
                result[f'{metric_name}_auc'] = y_values[0]
            else:
                auc = np.trapz(y_values, x=x_coords)
                x_range = max(x_coords) - min(x_coords)
                result[f'{metric_name}_auc'] = auc / x_range

    @staticmethod
    def _collect_array_data(data: pd.DataFrame, column: str) -> List[float]:
        """Collect and flatten array data from DataFrame column"""
        all_values = []
        for arr in data[column]:
            if isinstance(arr, np.ndarray):
                all_values.extend(arr.tolist())
            else:
                all_values.extend(arr)
        return all_values

    def _add_nan_metrics(self, result: Dict, level: int) -> None:
        """Add NaN values for all metrics at a given level"""
        metrics = ['repeatability_mean', 'localization_mean', 'response_ratio_mean']
        percentiles = ['loc_p50', 'loc_p95', 'loc_p99', 'resp_p50', 'resp_p95', 'resp_p99']

        for metric in metrics + percentiles:
            result[f'{metric}_{level}'] = np.nan

    def _format_aperture(self, row: pd.Series) -> str:
        """Format aperture parameter combination"""
        type_name = self.APERTURE_MAP[row['aperture_tp']]
        size = int(row['aperture_sz']) if row['aperture_sz'].is_integer() else row['aperture_sz']
        return f"{type_name}-{size}"

    def _format_window(self, row: pd.Series) -> str:
        """Format window parameter combination"""
        type_name = self.WINDOW_MAP[row['window_tp']]
        size = int(row['window_sz']) if row['window_sz'].is_integer() else row['window_sz']
        return f"{type_name}-{size}"

    def _format_param_string(self, params: pd.Series) -> str:
        """Format complete parameter string for combined analysis"""
        ap_type = self.APERTURE_MAP[params['aperture_tp']]
        win_type = self.WINDOW_MAP[params['window_tp']]

        return (f"k={params['k_val']:.3f}, "
                f"ap={ap_type}-{params['aperture_sz']}, "
                f"win={win_type}-{params['window_sz']}")

    def _print_best_isolated(self, results_df: pd.DataFrame) -> None:
        """Print best parameters for isolated analysis"""
        param_names = results_df['param_name'].unique()

        for metric in ['localization', 'repeatability']:
            auc_col = f'{metric}_auc'
            print(f"\nBest Parameter for {metric.title()} AUC:")

            for param in param_names:
                param_data = results_df[results_df['param_name'] == param]
                if not param_data.empty and not param_data[auc_col].isna().all():
                    best_idx = param_data[auc_col].idxmax()
                    best_value = param_data.loc[best_idx]
                    print(f"  {param}: {best_value['param_value']} "
                          f"(score: {best_value[auc_col]:.4f})")

    def _print_top_combined(self, results_df: pd.DataFrame, top_n: int) -> None:
        """Print top parameter combinations"""
        metrics = {
            'repeatability': 'Repeatability',
            'localization': 'Localization',
            'response_ratio': 'Response Ratio'
        }

        for metric_key, metric_name in metrics.items():
            auc_col = f'{metric_key}_auc'

            if auc_col not in results_df.columns or results_df[auc_col].isna().all():
                print(f"\nNo data available for {metric_name}")
                continue

            print(f"\nTop {top_n} Parameter Sets for {metric_name} Score:")
            top_params = results_df.nlargest(top_n, auc_col)

            for i, (_, row) in enumerate(top_params.iterrows(), 1):
                print(f"  {i}. {row['param_str']} (score: {row[auc_col]:.4f})")

    def _save_results(self, results_df: pd.DataFrame, filename: str) -> None:
        """Save results to parquet file"""
        filepath = f"results/distortion_{self.distortion_no}/{filename}"
        results_df.to_parquet(filepath, index=False)
        print(f"\nParameter analysis saved to {filepath}")


# Convenience functions for backward compatibility
def calculate_parameter_performance(df: pd.DataFrame, distortion_no: str) -> pd.DataFrame:
    """Calculate performance metrics for individual parameters (isolated analysis)"""
    analyzer = HarrisAnalyzer(df, distortion_no)
    return analyzer.analyze_parameters(mode='isolated')


def calculate_parameter_auc_confidence(df: pd.DataFrame, distortion_no: str) -> pd.DataFrame:
    """Calculate normalized AUC for all parameter combinations"""
    analyzer = HarrisAnalyzer(df, distortion_no)
    return analyzer.analyze_parameters(mode='combined')
