import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_recall_curve, f1_score
from sklearn.model_selection import train_test_split
import json
import os
from rules_baseline import apply_rules_to_dataframe

def load_data(train_path, test_path):
    """Loads training and testing data."""
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)
    return df_train, df_test

def feature_engineer(df):
    """Engineers features for the model."""
    # Convert timestamp to datetime
    df['timestamp'] = pd.to_datetime(df['timestamp_unix'], unit='s')

    # Time-based features
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Interaction features
    df['speed_x_pressure'] = df['speed'] * df['pressure_hpa']
    df['altitude_x_accuracy'] = df['altitude'] * df['horizontal_accuracy']

    # Categorical feature encoding (simple one-hot encoding)
    df = pd.get_dummies(df, columns=['wifi_bssid', 'cell_tower_id'], dummy_na=True)
    
    # For simplicity, we'll fill NaNs in numerical columns with the median
    for col in df.select_dtypes(include=np.number).columns:
        if df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Drop timestamp columns that are no longer needed
    if 'timestamp' in df.columns:
        df = df.drop(columns=['timestamp'])
    if 'timestamp_unix' in df.columns:
        df = df.drop(columns=['timestamp_unix'])
            
    return df

def train_model(X_train, y_train):
    """Trains a RandomForestClassifier."""
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def find_best_threshold(model, X_val, y_val):
    """Finds the best threshold from a PR curve based on F1 score."""
    y_scores = model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_scores)
    
    # Handle the case where thresholds array might be shorter
    f1_scores = [f1_score(y_val, y_scores >= t) for t in thresholds]
    
    best_f1_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_f1_idx]
    
    print(f"Best threshold found: {best_threshold:.4f} with F1-score: {f1_scores[best_f1_idx]:.4f}")
    
    return best_threshold

def main():
    """Main function to run the training and evaluation pipeline."""
    # Define paths
    data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    output_dir = os.path.join(os.path.dirname(__file__), '..')
    
    train_path = os.path.join(data_dir, 'train.csv')
    test_path = os.path.join(data_dir, 'test.csv')
    test_labels_path = os.path.join(data_dir, 'test_labels.csv')

    # Load data
    df_train, df_test = load_data(train_path, test_path)
    df_test_labels = pd.read_csv(test_labels_path)
    df_test_with_labels = pd.merge(df_test.copy(), df_test_labels, on='event_id')

    # Rename pressure column to be consistent
    if 'pressure' in df_train.columns:
        df_train.rename(columns={'pressure': 'pressure_hpa'}, inplace=True)
    if 'pressure' in df_test.columns:
        df_test.rename(columns={'pressure': 'pressure_hpa'}, inplace=True)
    if 'pressure' in df_test_with_labels.columns:
        df_test_with_labels.rename(columns={'pressure': 'pressure_hpa'}, inplace=True)

    # Feature Engineering
    df_train_featured = feature_engineer(df_train.drop(columns=['spoofed']))
    df_test_featured = feature_engineer(df_test.copy())
    
    # Align columns between train and test
    train_cols = set(df_train_featured.columns)
    test_cols = set(df_test_featured.columns)

    missing_in_test = list(train_cols - test_cols)
    for c in missing_in_test:
        df_test_featured[c] = 0

    missing_in_train = list(test_cols - train_cols)
    for c in missing_in_train:
        df_train_featured[c] = 0
            
    df_test_aligned = df_test_featured[df_train_featured.columns]

    # Define features and target
    features = [col for col in df_train_featured.columns if col not in ['event_id', 'timestamp', 'timestamp_unix', 'spoofed', 'wifi_bssid', 'ip_address', 'installation_id']]
    X = df_train_featured[features]
    y = df_train['spoofed']

    # Split training data for validation
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train model
    model = train_model(X_train, y_train)

    # Find best threshold
    threshold = find_best_threshold(model, X_val, y_val)

    # Predictions on test set
    X_test = df_test_aligned[features]
    test_scores_ml = model.predict_proba(X_test)[:, 1]
    test_flags_ml = (test_scores_ml >= threshold).astype(int)

    # --- Save results ---

    # 1. JSON output
    results_json = []
    for i, row in df_test.iterrows():
        results_json.append({
            'event_id': row['event_id'],
            'spoof_score_ml': test_scores_ml[i],
            'spoof_flag_ml': int(test_flags_ml[i])
        })
        
    with open(os.path.join(output_dir, 'results.json'), 'w') as f:
        json.dump(results_json, f, indent=4)

    # 2. CSVs for analysis
    # ML predictions
    df_test_preds_ml = df_test[['event_id']].copy()
    df_test_preds_ml['spoof_score_ml'] = test_scores_ml
    df_test_preds_ml['spoof_flag_ml'] = test_flags_ml
    df_test_preds_ml = pd.merge(df_test_preds_ml, df_test_labels, on='event_id')
    df_test_preds_ml.rename(columns={'spoofed': 'is_spoofed_ground_truth'}, inplace=True)
    df_test_preds_ml.to_csv(os.path.join(output_dir, 'ml_predictions.csv'), index=False)
    
    # Rules baseline predictions
    rules_flags = apply_rules_to_dataframe(df_test_with_labels.copy())
    df_test_preds_rules = df_test[['event_id']].copy()
    df_test_preds_rules['spoof_score_rules'] = rules_flags # Using flag as score
    df_test_preds_rules['spoof_flag_rules'] = rules_flags
    df_test_preds_rules = pd.merge(df_test_preds_rules, df_test_labels, on='event_id')
    df_test_preds_rules.rename(columns={'spoofed': 'is_spoofed_ground_truth'}, inplace=True)
    df_test_preds_rules.to_csv(os.path.join(output_dir, 'rules_predictions.csv'), index=False)

    # Hybrid predictions (simple average of scores)
    df_hybrid = pd.merge(df_test_preds_ml, df_test_preds_rules, on='event_id', suffixes=('_ml', '_rules'))
    df_hybrid['spoof_score_hybrid'] = (df_hybrid['spoof_score_ml'] + df_hybrid['spoof_score_rules']) / 2
    # Use a simple 0.5 threshold for the hybrid model for now
    df_hybrid['spoof_flag_hybrid'] = (df_hybrid['spoof_score_hybrid'] >= 0.5).astype(int)
    df_hybrid = df_hybrid[['event_id', 'spoof_score_hybrid', 'spoof_flag_hybrid', 'is_spoofed_ground_truth_ml']]
    df_hybrid.rename(columns={'is_spoofed_ground_truth_ml': 'is_spoofed_ground_truth'}, inplace=True)
    df_hybrid.to_csv(os.path.join(output_dir, 'hybrid_predictions.csv'), index=False)

    print("Training, evaluation, and prediction saving complete.")


if __name__ == '__main__':
    main()