import numpy as np
import pandas as pd


def calculate_parameter_auc_confidence(df, distortion_no):
    """
    Calculate normalized AUC and confidence levels for all parameter combinations

    Args:
        df: DataFrame with aggregated results from harris_wrapper
        distortion_no: Distortion type number

    Returns:
        DataFrame with normalized AUC scores and confidence levels
    """
    # Define parameter conversions
    aperture_type_map = {0: 'sobel', 1: 'prewitt', 2: 'scharr'}
    window_type_map = {0: 'uniform', 1: 'gaussian'}

    # Get all unique parameter combinations
    param_columns = ['k_val', 'aperture_tp', 'aperture_sz', 'window_tp', 'window_sz']
    unique_combinations = df[param_columns].drop_duplicates()

    # Results container
    results = []

    # Process each parameter combination
    for _, params in unique_combinations.iterrows():
        # Build filter condition
        filter_cond = pd.Series(True, index=df.index)
        for col in param_columns:
            filter_cond &= (df[col] == params[col])

        # Filter data for this parameter combination
        param_data = df[filter_cond]

        # Skip if no data
        if param_data.empty:
            continue

        # Create result dictionary with parameter values
        result = {
            'k_val': params['k_val'],
            'aperture_tp': aperture_type_map.get(params['aperture_tp'], str(params['aperture_tp'])),
            'aperture_sz': params['aperture_sz'],
            'window_tp': window_type_map.get(params['window_tp'], str(params['window_tp'])),
            'window_sz': params['window_sz'],
            'param_str': f"k={params['k_val']:.3f}, ap={aperture_type_map.get(params['aperture_tp'])}-{params['aperture_sz']}, win={window_type_map.get(params['window_tp'])}-{params['window_sz']}"
        }

        # Initialize metric arrays for AUC calculations
        repeatability_values = []
        localization_values = []
        response_ratio_values = []

        # Calculate metrics for each distortion level (1-5)
        for level in range(1, 6):
            level_data = param_data[param_data['dt_lv'] == level]

            if level_data.empty:
                # No data for this level
                for prefix in ['repeatability', 'localization', 'response_ratio']:
                    result[f'{prefix}_mean_{level}'] = float('nan')

                for pct in [50, 95, 99]:
                    result[f'loc_p{pct}_{level}'] = float('nan')
                    result[f'resp_p{pct}_{level}'] = float('nan')
                continue

            # 1. Process Repeatability
            mean_rep = level_data['mean_repeatability'].mean()
            result[f'repeatability_mean_{level}'] = mean_rep
            repeatability_values.append(mean_rep)

            # 2. Process Localization distances
            all_loc_distances = []
            for distances in level_data['loc_distances']:
                if isinstance(distances, np.ndarray):
                    all_loc_distances.extend(distances.tolist())
                else:
                    all_loc_distances.extend(distances)

            if all_loc_distances:
                mean_loc = np.mean(all_loc_distances)
                result[f'localization_mean_{level}'] = mean_loc
                localization_values.append(mean_loc)

                # Calculate percentiles
                result[f'loc_p50_{level}'] = np.percentile(all_loc_distances, 50)
                result[f'loc_p95_{level}'] = np.percentile(all_loc_distances, 95)
                result[f'loc_p99_{level}'] = np.percentile(all_loc_distances, 99)
            else:
                result[f'localization_mean_{level}'] = float('nan')
                result[f'loc_p50_{level}'] = float('nan')
                result[f'loc_p95_{level}'] = float('nan')
                result[f'loc_p99_{level}'] = float('nan')

            # 3. Process Response ratios
            all_resp_ratios = []
            for ratios in level_data['resp_ratios']:
                if isinstance(ratios, np.ndarray):
                    # Cap ratios at 1 as required
                    capped_ratios = np.minimum(ratios, 1).tolist()
                    all_resp_ratios.extend(capped_ratios)
                else:
                    # Cap ratios at 1 as required
                    capped_ratios = [min(r, 1) for r in ratios]
                    all_resp_ratios.extend(capped_ratios)

            if all_resp_ratios:
                mean_resp = np.mean(all_resp_ratios)
                result[f'response_ratio_mean_{level}'] = mean_resp
                response_ratio_values.append(mean_resp)

                # Calculate percentiles
                result[f'resp_p50_{level}'] = np.percentile(all_resp_ratios, 50)
                result[f'resp_p95_{level}'] = np.percentile(all_resp_ratios, 95)
                result[f'resp_p99_{level}'] = np.percentile(all_resp_ratios, 99)
            else:
                result[f'response_ratio_mean_{level}'] = float('nan')
                result[f'resp_p50_{level}'] = float('nan')
                result[f'resp_p95_{level}'] = float('nan')
                result[f'resp_p99_{level}'] = float('nan')

        # Calculate normalized AUC scores

        # 1. Repeatability AUC (already normalized as is)
        if repeatability_values:
            # Create arrays with valid values
            valid_indices = [i for i, v in enumerate(repeatability_values) if not np.isnan(v)]
            x_coords = [i+1 for i in valid_indices]  # Distortion levels (1-based)
            y_values = [repeatability_values[i] for i in valid_indices]

            if y_values:
                # Calculate AUC using trapz and normalize by x-range
                auc = np.trapz(y_values, x=x_coords)
                x_range = max(x_coords) - min(x_coords)
                result['repeatability_auc'] = auc / x_range if x_range > 0 else np.mean(y_values)
            else:
                result['repeatability_auc'] = float('nan')
        else:
            result['repeatability_auc'] = float('nan')

        # 2. Localization AUC (normalize using 1/(1+d))
        if localization_values:
            # Apply normalization formula 1/(1+d) and filter out NaNs
            valid_indices = [i for i, v in enumerate(localization_values) if not np.isnan(v)]
            x_coords = [i+1 for i in valid_indices]  # Distortion levels (1-based)
            y_values = [1/(1+localization_values[i]) for i in valid_indices]

            if y_values:
                # Calculate AUC using trapz and normalize by x-range
                auc = np.trapz(y_values, x=x_coords)
                x_range = max(x_coords) - min(x_coords)
                result['localization_auc'] = auc / x_range if x_range > 0 else np.mean(y_values)
            else:
                result['localization_auc'] = float('nan')
        else:
            result['localization_auc'] = float('nan')

        # 3. Response Ratio AUC (capping at 1)
        if response_ratio_values:
            # Filter out NaNs
            valid_indices = [i for i, v in enumerate(response_ratio_values) if not np.isnan(v)]
            x_coords = [i+1 for i in valid_indices]  # Distortion levels (1-based)
            y_values = [response_ratio_values[i] for i in valid_indices]

            if y_values:
                # Calculate AUC using trapz and normalize by x-range
                auc = np.trapz(y_values, x=x_coords)
                x_range = max(x_coords) - min(x_coords)
                result['response_ratio_auc'] = auc / x_range if x_range > 0 else np.mean(y_values)
            else:
                result['response_ratio_auc'] = float('nan')
        else:
            result['response_ratio_auc'] = float('nan')

        # Add to results
        results.append(result)

    # Convert to DataFrame
    results_df = pd.DataFrame(results) if results else pd.DataFrame()

    # Print top parameter combinations
    if not results_df.empty:
        print_top_parameters(results_df)

        # Save to parquet
        results_df.to_parquet(f"results/distortion_{distortion_no}/harris_param_combined.parquet")
        print(f"\nParameter analysis saved to results/distortion_{distortion_no}/harris_param_combined.parquet")
    else:
        print("\nNo results to analyze")

    return results_df


def print_top_parameters(results_df, top_n=3):
    """Print the top parameter combinations for each metric"""
    metrics = {
        'repeatability': 'Repeatability',
        'localization': 'Localization',
        'response_ratio': 'Response Ratio'
    }

    for metric_key, metric_name in metrics.items():
        auc_col = f'{metric_key}_auc'

        # Check if column exists and has non-NaN values
        if auc_col not in results_df.columns or results_df[auc_col].isna().all():
            print(f"\nNo data available for {metric_name}")
            continue

        print(f"\nTop {top_n} Parameter Sets for {metric_name} Score:")

        # Get top parameter combinations
        top_params = results_df.nlargest(top_n, auc_col)

        # Print each combination
        for i, (_, row) in enumerate(top_params.iterrows(), 1):
            print(f"  {i}. {row['param_str']} (score: {row[auc_col]:.4f})")
