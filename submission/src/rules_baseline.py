
import pandas as pd
import numpy as np

# --- Rule Constants ---
# These can be tuned for better performance
SPEED_THRESHOLD_MPS = 343  # Speed of sound, a hard limit for teleportation
PRESSURE_DEVIATION_THRESHOLD = 15  # Max allowed hPa deviation for a given altitude
FROZEN_LOCATION_SPEED_THRESHOLD = 1.0 # Min speed to be considered "moving"

def _haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance in meters between two points
    on the earth, specified in decimal degrees.
    Internal use function.
    """
    # Return NaN if any input is NaN
    if np.any(pd.isna([lat1, lon1, lat2, lon2])):
        return np.nan
        
    R = 6371e3  # Radius of Earth in meters
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)

    a = np.sin(delta_phi / 2)**2 + np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2)**2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    return R * c

def apply_rules_to_dataframe(df_input):
    """
    Applies a set of heuristic rules to a dataframe to detect spoofing.

    This function is designed to be imported into other scripts. It takes a
    dataframe, prepares it for sequential analysis, and applies several
    rules to generate a prediction for each event.

    Args:
        df_input (pd.DataFrame): The input dataframe, expected to have the
                                 schema from generate_data.py.

    Returns:
        pd.Series: A series of predictions (1 for spoofed, 0 for normal),
                   aligned with the index of the input dataframe.
    """
    df = df_input.copy()

    # --- 1. Data Preparation for Sequential Analysis ---
    # Sort by device and time to correctly analyze journeys
    df = df.sort_values(by=['installation_id', 'timestamp_unix'])
    
    # Get previous event data for each installation
    grouped = df.groupby('installation_id')
    for col in ['latitude', 'longitude', 'timestamp_unix']:
        df[f'prev_{col}'] = grouped[col].shift(1)
        
    # Calculate distance and speed from the previous point
    df['dist_from_prev'] = df.apply(
        lambda row: _haversine_distance(row['latitude'], row['longitude'], row['prev_latitude'], row['prev_longitude']), 
        axis=1
    )
    time_delta = df['timestamp_unix'] - df['prev_timestamp_unix']
    # Replace 0 time delta with NaN to avoid division by zero
    df['speed_from_prev_mps'] = df['dist_from_prev'] / time_delta.replace(0, np.nan)

    # --- 2. Apply Rules ---
    # Start with all predictions as 0 (normal)
    df['prediction'] = 0

    # Rule 1: Direct Mock Location Flag
    # The most direct evidence of spoofing from the OS.
    df.loc[df['mock_location_enabled'] == True, 'prediction'] = 1

    # Rule 2: Impossible Speed (Teleportation)
    # If calculated speed between points exceeds the threshold.
    df.loc[df['speed_from_prev_mps'] > SPEED_THRESHOLD_MPS, 'prediction'] = 1

    # Rule 3: Perfect Accuracy Anomaly
    # Simple bots or emulators often report perfect accuracy.
    df.loc[(df['horizontal_accuracy'] == 1.0), 'prediction'] = 1

    # Rule 4: Frozen Location
    # The device reports it is moving, but its GPS coordinates are static.
    is_frozen = (df['dist_from_prev'] == 0) & (df['speed'] > FROZEN_LOCATION_SPEED_THRESHOLD)
    df.loc[is_frozen, 'prediction'] = 1

    # Rule 5: Altitude-Pressure Mismatch
    # Checks for physical consistency between altitude and barometric pressure.
    # Formula: Pressure decreases by ~1 hPa per 8.3 meters of altitude gain.
    expected_pressure = 1013.25 - (df['altitude'] / 8.3)
    pressure_deviation = np.abs(df['pressure_hpa'] - expected_pressure)
    df.loc[pressure_deviation > PRESSURE_DEVIATION_THRESHOLD, 'prediction'] = 1

    # --- 3. Return Predictions ---
    # Ensure the output series is aligned with the original dataframe's index
    return df['prediction'].reindex(df_input.index)

# This file is intended to be used as a module.
# The main execution block is left empty.
if __name__ == '__main__':
    print("This script is a module and is meant to be imported.")
    print("Example usage:")
    print("from rules_baseline import apply_rules_to_dataframe")
    print("predictions = apply_rules_to_dataframe(my_dataframe)")
    pass
