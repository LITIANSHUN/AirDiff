# Code claim: Read data—use pd.read_csv to load .log file and explicitly set column names.
# Data cleaning:
#   - Use dropna to remove rows that contain missing values.
#   - Ensure all numeric fields are cast to float via astype(float).
# Data transformation:
#   - Convert lat and lon from degrees to radians with math.radians.
#   - Convert simt to a timestamp using pd.to_datetime, storing it in a new column timestamp.
# Data filtering:
#   - Keep only records whose type equals M600 via boolean indexing.
#   - Keep only records whose alt is greater than 0 via boolean indexing.
# Save processed data: use to_csv to write the cleaned dataset to a .csv file.
# Error handling: catch and report FileNotFoundError, EmptyDataError, and any other unexpected exceptions.
# Usage:
#   - Replace input_file and output_file with the actual file paths.
#   - Run the script; the processed data will be saved to the specified output file.

import pandas as pd
import numpy as np
import math

def preprocess_log_file(input_file, output_file):
    try:
        # Read data
        df = pd.read_csv(input_file, header=None, names=[
            'simt', 'id', 'type', 'lat', 'lon', 'alt', 'tas', 'cas', 'vs', 'gs', 
            'distflown', 'Temp', 'trk', 'hdg', 'p', 'rho', 'thrust', 'drag', 
            'phase', 'fuelflow'
        ])

        # Data cleaning: drop any rows with missing values
        df.dropna(inplace=True)

        # Ensure all numeric columns are of float type
        numeric_columns = [
            'simt', 'lat', 'lon', 'alt', 'tas', 'cas', 'vs', 'gs', 
            'distflown', 'Temp', 'trk', 'hdg', 'p', 'rho', 'thrust', 
            'drag', 'fuelflow'
        ]
        df[numeric_columns] = df[numeric_columns].astype(float)

        # Data transformation: convert lat and lon from degrees to radians
        df['lat'] = df['lat'].apply(math.radians)
        df['lon'] = df['lon'].apply(math.radians)

        # Add a new column 'timestamp' by converting simt (assumed to be in seconds) to datetime
        df['timestamp'] = pd.to_datetime(df['simt'], unit='s')

        # Data filtering: keep only records with type == 'M600'
        df = df[df['type'] == 'M600']

        # Data filtering: keep only records with alt > 0
        df = df[df['alt'] > 0]

        # Save the processed data
        df.to_csv(output_file, index=False)

        print(f"Data preprocessing completed; results saved to {output_file}")

    except FileNotFoundError:
        print(f"Error: file {input_file} not found.")
    except pd.errors.EmptyDataError:
        print(f"Error: file {input_file} is empty or has incorrect format.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


# Example usage
if __name__ == "__main__":
    # Replace input_file and output_file with actual file paths
    input_file = 'data.log'      # Path to input file
    output_file = 'processed_data.csv'  # Path to output file
    preprocess_log_file(input_file, output_file)
