import numpy as np
import pandas as pd
import glob
import os


def load_harris_metrics(distortion_type: str) -> pd.DataFrame:
    """
    Collect Harris performance metrics of a distortion type
    Returns:
        Merged DataFrame containing all metrics
    """
    # Construct file paths
    file_pattern = os.path.join(f"results/distortion_{distortion_type}", "harris_*.csv")
    file_paths = glob.glob(file_pattern)

    # Load and validate dataframes
    valid_dataframes = []
    row_num = 0
    file_num = 0
    for path in file_paths:
        df = pd.read_csv(path)

        # Skip files with invalid dt_no values
        if not ('dt_no' in df.columns and all(df['dt_no'] == int(distortion_type))):
            print(f"Warning: File {path} has invalid dt_no values, skipping")
        else:
            row_num += len(df)
            file_num += 1
            valid_dataframes.append(df)
    # Merge valid dataframes
    print(f"Collected {row_num} data rows across {file_num} file")
    return pd.concat(valid_dataframes, ignore_index=True) if valid_dataframes else pd.DataFrame()


def analyze_parameter_performance(data, distortion_type):
    """
    Analyzes parameter performance based on repeatability across distortion levels - Mikolajczyk&Schmid(2005)
    Returns:
         Dataframe of performance score for each parameter value
    """
    print(f"\nMost Robust Parameter Values for distortion #{int(distortion_type)}:")

    # Initialize result dataframes
    results_df = []
    # Iterate through implementation parameters
    parameters = ['window_sz', 'windows_tp', 'aperture_sz', 'aperture_tp', 'k_val']
    for param_name in parameters:
        param_values = sorted(data[param_name].unique())
        best_value = -1
        best_score = 0
        # Iterate through parameter values
        for param_val in param_values:
            subset = data[data[param_name] == param_val]

            # Initialize row
            result_row = {
                'parameter': param_name,
                'value': param_val
            }

            # Calculate mean repeatability for each distortion level
            level_means = []
            for level in range(1, 6):
                level_data = subset[subset['dt_lv'] == level]
                mean_repeatability = level_data['repeatability'].mean()
                result_row[f'level_{level}'] = mean_repeatability
                level_means.append(mean_repeatability)

            # Calculate AUC using trapezoid rule (normalized for 5 level)
            auc_repeatability = np.trapezoid(level_means, dx=1) / 4
            result_row['robustness_auc'] = auc_repeatability

            # Track best parameter value
            if auc_repeatability > best_score:
                best_score = auc_repeatability
                best_value = param_val

            results_df.append(result_row)

        # Print the best parameter value
        if param_name == 'windows_tp':
            value_str ={0: 'uniform', 1: 'gaussian'}.get(best_value, f"Unknown aperture {best_value}")
        elif param_name == 'aperture_tp':
            value_str = {0: 'sobel', 1: 'prewitt', 2: 'scharr'}.get(best_value, f"Unknown window {best_value}")
        else:
            value_str = best_value

        print(f"{param_name}: {value_str} (AUC Score: {best_score:.4f})")

    # Convert to dataframe
    results_df = pd.DataFrame(results_df)

    # Ensure directory exists
    os.makedirs(f"results/distortion_{distortion_type}_analysis", exist_ok=True)

    # Save to CSV
    csv_path = f"results/distortion_{distortion_type}_analysis/harris_AUC.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nSaved parameter AUC scores to {csv_path}")

    return results_df
