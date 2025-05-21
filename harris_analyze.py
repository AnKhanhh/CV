import numpy as np
import pandas as pd


def calculate_parameter_performance(df, distortion_no):
    """
    Calculate performance metrics for individual parameters and meaningful combinations
    Args:
        df: DataFrame with aggregated results from harris_wrapper
        distortion_no: Distortion type number
    Returns:
        Single DataFrame containing all parameter results
    """
    # Define all parameters to analyze
    parameters = [
        {'param_name': 'k_val', 'params': ['k_val']},
        {'param_name': 'aperture', 'params': ['aperture_tp', 'aperture_sz']},
        {'param_name': 'window', 'params': ['window_tp', 'window_sz']}
    ]

    # Single results container
    all_results = []

    # Process all parameters (individual and combinations) using the same flow
    for param_config in parameters:
        param_name = param_config['param_name']
        param_cols = param_config['params']

        # Get unique parameter values or combinations
        unique_values = df[param_cols].drop_duplicates()

        for _, value_row in unique_values.iterrows():
            # Build filter condition
            filter_cond = pd.Series(True, index=df.index)
            for col in param_cols:
                filter_cond &= (df[col] == value_row[col])

            # Create parameter value label
            if len(param_cols) == 1:
                # Individual parameter - convert to string to ensure type compatibility
                param_value = str(value_row[param_cols[0]])
            else:
                # Format combination value for better readability
                parts = []
                for col in param_cols:
                    if col == 'aperture_tp':
                        value_name = {0: 'sobel', 1: 'prewitt', 2: 'scharr'}[value_row[col]]
                    elif col == 'window_tp':
                        value_name = {0: 'uniform', 1: 'gaussian'}[value_row[col]]
                    else:
                        val = value_row[col]
                        value_name = str(int(val) if isinstance(val, float) and val.is_integer() else val)
                    parts.append(value_name)
                param_value = "-".join(parts)

            # Filter data for this parameter value/combination
            filtered_data = df[filter_cond]

            # Skip if no data (shouldn't happen but good to check)
            if filtered_data.empty:
                continue

            # Create result dictionary
            result = {
                'param_name': param_name,
                'param_value': param_value  # Always a string for type compatibility
            }

            # Calculate metrics for this parameter value/combination
            calculate_level_metrics(filtered_data, result)

            # Add to results
            all_results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(all_results) if all_results else pd.DataFrame()

    # Print best parameter values
    if not results_df.empty:
        print_best_parameters(results_df)
        results_df.to_csv(f"results/distortion_{distortion_no}/harris_params.csv", index=False)
        print(f"\nParameter analysis saved to results/distortion_{distortion_no}/harris_params.csv")

    return results_df


def calculate_level_metrics(data, result):
    """Calculate metrics for each distortion level and store in result dict"""
    repeatability_values = []
    localization_values = []

    for level in range(1, 6):  # 5 distortion levels
        level_data = data[data['dt_lv'] == level]

        if level_data.empty:
            # No data for this level
            result[f'mean_repeatability_{level}'] = float('nan')
            result[f'mean_localization_{level}'] = float('nan')
            result[f'p50_local_{level}'] = float('nan')
            result[f'p95_local_{level}'] = float('nan')
            result[f'p99_local_{level}'] = float('nan')
            continue

        # Get mean repeatability for this level
        mean_rep = level_data['mean_repeatability'].mean()
        result[f'mean_repeatability_{level}'] = mean_rep
        repeatability_values.append(mean_rep)

        # Combine all localization distances for this level
        all_distances = []
        for distances in level_data['loc_distances']:
            if isinstance(distances, np.ndarray):
                all_distances.extend(distances.tolist())
            else:
                all_distances.extend(distances)

        if all_distances:
            # Calculate mean localization
            mean_loc = np.mean(all_distances)
            result[f'mean_localization_{level}'] = mean_loc
            localization_values.append(mean_loc)

            # Calculate percentiles
            result[f'p50_local_{level}'] = np.percentile(all_distances, 50)
            result[f'p95_local_{level}'] = np.percentile(all_distances, 95)
            result[f'p99_local_{level}'] = np.percentile(all_distances, 99)
        else:
            result[f'mean_localization_{level}'] = float('nan')
            result[f'p50_local_{level}'] = float('nan')
            result[f'p95_local_{level}'] = float('nan')
            result[f'p99_local_{level}'] = float('nan')

    # Calculate AUC across all distortion levels
    if repeatability_values:
        result['repeatability_auc'] = np.mean(repeatability_values)
    else:
        result['repeatability_auc'] = float('nan')

    if localization_values:
        # Calculate 1/(1+d) for each level and take mean
        norm_loc_values = [1 / (1 + d) for d in localization_values if not np.isnan(d)]
        result['localization_auc'] = np.mean(norm_loc_values) if norm_loc_values else float('nan')
    else:
        result['localization_auc'] = float('nan')


def print_best_parameters(results_df):
    """Print the best parameter values for localization and repeatability"""
    if results_df.empty:
        print("No results to analyze")
        return

    # Get unique parameter names
    param_names = results_df['param_name'].unique()

    print("\nBest Parameter for Localization AUC:")
    for param in param_names:
        param_data = results_df[results_df['param_name'] == param]
        if not param_data.empty and not param_data['localization_auc'].isna().all():
            best_idx = param_data['localization_auc'].idxmax()
            best_value = param_data.loc[best_idx]
            print(f"  {param}: {best_value['param_value']} (score: {best_value['localization_auc']:.4f})")

    print("\nBest Parameter for Repeatability AUC:")
    for param in param_names:
        param_data = results_df[results_df['param_name'] == param]
        if not param_data.empty and not param_data['repeatability_auc'].isna().all():
            best_idx = param_data['repeatability_auc'].idxmax()
            best_value = param_data.loc[best_idx]
            print(f"  {param}: {best_value['param_value']} (score: {best_value['repeatability_auc']:.4f})")
