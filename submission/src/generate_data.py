import os
import uuid
import random
import numpy as np
import pandas as pd

# --- Configuration ---
N_TRAIN_ROWS = 10000
N_TEST_ROWS = 3000
SPOOF_RATE_MIN = 0.15
SPOOF_RATE_MAX = 0.30
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))

# Constants for simulation
BASE_LAT, BASE_LON = 40.7128, -74.0060  # New York City
WALKING_SPEED_MPS = 1.4
DRIVING_SPEED_MPS = 15.0 # Approx 54 km/h or 33 mph
MPS_TO_DEG_APPROX = 1 / 111320

# Pre-defined realistic values for environmental anchors
WIFI_BSSIDS = [f"0a:1b:2c:3d:4e:{i:02x}" for i in range(10)]
CELL_TOWER_IDS = [f"420-55-{random.randint(1000, 9999)}-{random.randint(1000, 9999)}" for _ in range(20)]


def generate_base_event(timestamp, lat, lon):
    """Generates a single event with realistic sensor noise."""
    base_altitude = np.random.normal(50, 10)
    return {
        "timestamp_unix": int(timestamp),
        "latitude": lat + np.random.normal(0, 0.00001),
        "longitude": lon + np.random.normal(0, 0.00001),
        "horizontal_accuracy": max(2.0, np.random.normal(10, 5)),
        "vertical_accuracy": max(3.0, np.random.normal(15, 8)),
        "altitude": max(10, base_altitude),
        "pressure_hpa": 1013.25 - (base_altitude / 8.3) + np.random.normal(0, 0.5),
        "ambient_light_lux": max(0, np.random.normal(200, 50)),
        "num_satellites": random.randint(8, 20),
        "device_is_charging": random.random() < 0.2,
        "mock_location_enabled": False,
    }

def simulate_stationary_or_walking_journey(start_time, start_lat, start_lon, num_events):
    """Simulates a normal user journey (walking or stationary)."""
    events = []
    lat, lon = start_lat, start_lon
    is_stationary = random.random() < 0.5
    speed = 0 if is_stationary else WALKING_SPEED_MPS
    wifi_bssid = random.choice(WIFI_BSSIDS) if is_stationary and random.random() < 0.8 else None
    cell_tower_id = random.choice(CELL_TOWER_IDS)
    
    for i in range(num_events):
        timestamp = start_time + i * 10
        event = generate_base_event(timestamp, lat, lon)
        event.update({
            "speed": max(0, speed + np.random.normal(0, 0.2)) if not is_stationary else 0,
            "bearing": np.random.uniform(0, 360) if not is_stationary else 0,
            "wifi_bssid": wifi_bssid,
            "cell_tower_id": cell_tower_id
        })
        events.append(event)
        if not is_stationary:
            direction = np.random.uniform(0, 360)
            lat += speed * 10 * MPS_TO_DEG_APPROX * np.cos(np.radians(direction))
            lon += speed * 10 * MPS_TO_DEG_APPROX * np.sin(np.radians(direction))
    return events

def simulate_driving_journey(start_time, start_lat, start_lon, num_events):
    """Simulates a normal user journey while driving."""
    events = []
    lat, lon = start_lat, start_lon
    speed = DRIVING_SPEED_MPS
    # Driving implies a more consistent direction
    base_bearing = np.random.uniform(0, 360)
    # Cell towers change while driving
    cell_tower = random.choice(CELL_TOWER_IDS)
    
    for i in range(num_events):
        timestamp = start_time + i * 10 # 10s interval
        event = generate_base_event(timestamp, lat, lon)
        
        # Change cell tower every 5-10 events
        if i > 0 and i % random.randint(5, 10) == 0:
            cell_tower = random.choice(CELL_TOWER_IDS)
            
        event.update({
            "speed": max(0, speed + np.random.normal(0, 2)),
            "bearing": base_bearing + np.random.normal(0, 5), # Keep bearing somewhat stable
            "wifi_bssid": None, # Unlikely to be on WiFi while driving
            "cell_tower_id": cell_tower
        })
        events.append(event)
        
        # Move the user
        direction = base_bearing + np.random.normal(0, 5)
        lat += speed * 10 * MPS_TO_DEG_APPROX * np.cos(np.radians(direction))
        lon += speed * 10 * MPS_TO_DEG_APPROX * np.sin(np.radians(direction))
        
    return events

def simulate_normal_journey(start_time, start_lat, start_lon, num_events):
    """Dispatcher for different types of normal journeys."""
    if random.random() < 0.3: # 30% chance of driving
        return simulate_driving_journey(start_time, start_lat, start_lon, num_events)
    else: # 70% chance of walking or stationary
        return simulate_stationary_or_walking_journey(start_time, start_lat, start_lon, num_events)

def simulate_spoofed_journey(start_time, start_lat, start_lon, num_events):
    """Simulates a journey with one of five spoofing attack types."""
    spoof_type = random.choice([
        "mock_provider", "teleport", "bot_replay", "frozen_location", "sensor_mismatch"
    ])
    
    # Use a normal journey as a base to modify
    events = simulate_normal_journey(start_time, start_lat, start_lon, num_events)
    
    if spoof_type == "mock_provider":
        for event in events: event["mock_location_enabled"] = True
        return events

    if spoof_type == "teleport":
        if num_events < 2: return events
        teleport_index = random.randint(1, num_events - 1)
        pre_teleport_wifi = events[teleport_index-1]["wifi_bssid"]
        pre_teleport_cell = events[teleport_index-1]["cell_tower_id"]
        
        events[teleport_index]["latitude"] += 0.1
        events[teleport_index]["longitude"] += 0.1
        events[teleport_index]["wifi_bssid"] = pre_teleport_wifi
        events[teleport_index]["cell_tower_id"] = pre_teleport_cell
        events[teleport_index]["speed"] = 1500 + np.random.uniform(100, 200)
        return events

    if spoof_type == "bot_replay":
        for event in events:
            event["horizontal_accuracy"] = 1.0; event["vertical_accuracy"] = 1.0
            event["num_satellites"] = 25; event["speed"] = DRIVING_SPEED_MPS
            event["bearing"] = 45
        return events

    if spoof_type == "frozen_location":
        frozen_lat, frozen_lon = events[0]["latitude"], events[0]["longitude"]
        for event in events:
            event["latitude"], event["longitude"] = frozen_lat, frozen_lon
            event["speed"] = WALKING_SPEED_MPS + np.random.normal(0, 0.2)
        return events

    if spoof_type == "sensor_mismatch":
        base_pressure = events[0]["pressure_hpa"]
        for i, event in enumerate(events):
            event["altitude"] = 50 + 25 * np.sin(i / 5)
            event["pressure_hpa"] = base_pressure + np.random.normal(0, 0.1)
        return events
        
    return events

def generate_dataset(num_rows, spoof_rate):
    """Generates the full dataset with multiple journeys."""
    all_events = []
    
    n_journeys = num_rows // 25
    n_spoofed_journeys = int(n_journeys * spoof_rate)
    n_normal_journeys = n_journeys - n_spoofed_journeys

    for _ in range(n_normal_journeys):
        journey_len = random.randint(15, 40)
        lat, lon = BASE_LAT + np.random.uniform(-0.1, 0.1), BASE_LON + np.random.uniform(-0.1, 0.1)
        events = simulate_normal_journey(pd.Timestamp.now().timestamp(), lat, lon, journey_len)
        for e in events: e["spoofed"] = 0
        all_events.extend(events)

    for _ in range(n_spoofed_journeys):
        journey_len = random.randint(15, 40)
        lat, lon = BASE_LAT + np.random.uniform(-0.1, 0.1), BASE_LON + np.random.uniform(-0.1, 0.1)
        events = simulate_spoofed_journey(pd.Timestamp.now().timestamp(), lat, lon, journey_len)
        for e in events: e["spoofed"] = 1
        all_events.extend(events)
        
    df = pd.DataFrame(all_events)
    
    unique_install_ids = [uuid.uuid4() for _ in range(n_journeys)]
    df['installation_id'] = [random.choice(unique_install_ids) for _ in range(len(df))]
    df['event_id'] = [uuid.uuid4() for _ in range(len(df))]

    cols = [
        'event_id', 'installation_id', 'latitude', 'longitude', 'timestamp_unix',
        'horizontal_accuracy', 'altitude', 'speed', 'bearing', 'pressure_hpa',
        'mock_location_enabled', 'device_is_charging', 'wifi_bssid',
        'cell_tower_id', 'num_satellites', 'vertical_accuracy',
        'ambient_light_lux', 'spoofed'
    ]
    df = df.reindex(columns=cols)
    
    return df.sample(frac=1).reset_index(drop=True).head(num_rows)

def main():
    print("Starting data generation...")
    if not os.path.exists(OUTPUT_DIR):
        print(f"Creating output directory: {OUTPUT_DIR}")
        os.makedirs(OUTPUT_DIR)

    print(f"Generating training data (~{N_TRAIN_ROWS} rows)...")
    train_spoof_rate = np.random.uniform(SPOOF_RATE_MIN, SPOOF_RATE_MAX)
    train_df = generate_dataset(N_TRAIN_ROWS, train_spoof_rate)
    train_path = os.path.join(OUTPUT_DIR, "train.csv")
    train_df.to_csv(train_path, index=False)
    
    actual_spoof_rate = train_df['spoofed'].mean()
    print(f"Successfully generated {train_path} with {len(train_df)} rows.")
    print(f"Reported training spoof rate: {actual_spoof_rate:.4f}")

    print(f"\nGenerating test data (~{N_TEST_ROWS} rows)...")
    test_spoof_rate = np.random.uniform(0.1, 0.4) 
    test_df = generate_dataset(N_TEST_ROWS, test_spoof_rate)
    
    test_labels = test_df[['event_id', 'spoofed']]
    test_unlabeled_df = test_df.drop(columns=['spoofed'])
    test_path = os.path.join(OUTPUT_DIR, "test.csv")
    labels_path = os.path.join(OUTPUT_DIR, "test_labels.csv")
    test_unlabeled_df.to_csv(test_path, index=False)
    test_labels.to_csv(labels_path, index=False)

    print(f"Successfully generated {test_path} (unlabeled).")
    print(f"Successfully generated {labels_path} (ground truth).")
    print("\nData generation complete. You can now run this script.")

if __name__ == "__main__":
    main()