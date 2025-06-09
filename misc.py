import pandas as pd
import numpy as np
from pandasgui import show

FILE_PATH = "results/distortion_01/fast_param_combined.parquet"
NUM_ROWS = 3  # 0 to read all rows


def print_parquet_schema(file_path=None):
    """
    Print schema information for a parquet file.

    Args:
        file_path (str, optional): Path to parquet file. Uses global FILE_PATH if None.
    """
    path = file_path or globals().get('FILE_PATH')
    if not path:
        raise ValueError("No file path provided and FILE_PATH not found in global namespace")

    # Read parquet metadata without loading full data
    df = pd.read_parquet(path, engine='pyarrow')

    print(f"Schema for: {path}")
    print(f"Shape: {df.shape}")
    print("\nColumn Information:")
    print("-" * 50)

    for col in df.columns:
        dtype = df[col].dtype
        null_count = df[col].isnull().sum()
        print(f"{col:<30} {str(dtype):<15} (nulls: {null_count})")


def open_pq():
    # Read the parquet file
    print(f"Reading file: {FILE_PATH}")
    df = pd.read_parquet(FILE_PATH)

    # Select specified number of rows if NUM_ROWS > 0
    if NUM_ROWS > 0:
        df = df.head(NUM_ROWS)
        print(f"Displaying first {NUM_ROWS} rows")
    else:
        print(f"Displaying all {len(df)} rows")

    # Handle array columns for display
    # Convert numpy arrays to lists for better display in pandasgui
    for col in df.columns:
        if len(df) > 0 and isinstance(df[col].iloc[0], np.ndarray):
            df[col] = df[col].apply(lambda x: x.tolist())

    # Launch pandasgui
    print("Launching pandas GUI...")
    gui = show(df, settings={'block': True})

    return gui


if __name__ == "__main__":
    # print_parquet_schema()
    print_parquet_schema()
