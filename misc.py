import pandas as pd
import numpy as np
from pandasgui import show


def print_progress_bar(iteration, total, bar_length=50):
    """
    Print a progress bar that updates in place.
    """
    # Calculate progress and filled positions
    progress = iteration / total
    filled_length = int(bar_length * progress)

    # Create the bar
    bar = '=' * filled_length + ' ' * (bar_length - filled_length)

    # Print the progress bar
    print(f'\rIteration {iteration}/{total}: [{bar}]', end='')

    # Print a newline when complete
    if iteration == total:
        print()


def open_pq():
    # Static variables to modify
    FILE_PATH = "results/distortion_01/harris_param_combined.parquet"
    NUM_ROWS = 10  # Number of rows to read (0 for all rows)

    try:
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

    except ImportError:
        print("pandasgui not installed. Please install it with:")
        print("pip install pandasgui")
        return None
    except Exception as e:
        print(f"Error: {e}")
        return None


if __name__ == "__main__":
    open_pq()
