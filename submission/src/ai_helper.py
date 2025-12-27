import pandas as pd
import json
import argparse
import os
import sys
import time

# Since we cannot make live API calls, we will mock the functionality.
# If you have an API key, you can uncomment the google.generativeai lines.
import google.genai as genai
MODEL_ID = "gemini-2.5-flash"

def generate_mock_explanation(row):
    """
    Generates a deterministic, human-readable explanation for a flagged event
    based on its features. This function serves as a mock for a real LLM call.
    """
    reasons = []
    # Rule 1: Check for emulators or rooted devices
    if row.get('is_emulator', False):
        reasons.append("the event originated from an Android emulator")
    elif row.get('is_rooted', False):
        reasons.append("the device is rooted, which makes it easier to install spoofing software")

    # Rule 2: Check for suspicious speed (e.g., > 100 m/s is > 360 km/h)
    if row.get('speed', 0) > 100:
        reasons.append(f"the reported speed of {row['speed']:.0f} m/s is unrealistic for a ground vehicle")

    # Rule 3: Check for suspicious altitude
    if row.get('altitude', 0) > 8000:
        reasons.append(f"the altitude of {row['altitude']:.0f} meters is at commercial flight level, which is suspicious if not tracked over time")

    # Rule 4: Check for low accuracy which might be a sign of poor signal or intentional obfuscation
    if row.get('accuracy', 0) > 1000:
        reasons.append(f"the location accuracy of {row['accuracy']:.0f} meters is very low, suggesting a poor GPS signal or obfuscation")
    
    # Default explanation if no specific rules are triggered
    if not reasons:
        return "This event was flagged by the model due to a subtle combination of feature values that deviate from typical, non-spoofed behavior patterns observed in the training data."

    # Combine reasons into a single sentence
    explanation = "This event is likely spoofed because " + " and ".join(reasons) + "."
    return explanation

def main():
    """
    Main function to load predictions, sample flagged events, and generate explanations.
    """
    parser = argparse.ArgumentParser(description="Generate natural language explanations for flagged events.")
    parser.add_argument(
        "--frac",
        type=float,
        default=0.1,
        help="Fraction of spoofed events to sample for explanation (e.g., 0.1 for 10%%)."
    )
    args = parser.parse_args()

    if not 0 < args.frac <= 1:
        print("Error: --frac must be between 0 and 1.", file=sys.stderr)
        sys.exit(1)

    # Define paths relative to the script's location in submission/src/
    base_path = os.path.dirname(__file__)
    predictions_path = os.path.join(base_path, '..', 'ml_predictions.csv')
    features_path = os.path.join(base_path, '..', 'data', 'test.csv')
    results_path = os.path.join(base_path, '..', 'results_sample.json')

    # --- Step 1: Load data ---
    try:
        df_preds = pd.read_csv(predictions_path)
        df_features = pd.read_csv(features_path)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}. Ensure 'ml_predictions.csv' and 'data/test.csv' exist.", file=sys.stderr)
        sys.exit(1)

    # --- Step 2: Filter for flagged events and sample ---
    df_full = pd.merge(df_preds, df_features, on='event_id')
    df_flagged = df_full[df_full['spoof_flag_ml'] == 1].copy()
    
    if df_flagged.empty:
        print("No events flagged as spoofed. Nothing to explain.")
        return
        
    df_sample = df_flagged.sample(frac=min(args.frac, 1.0), random_state=42)
    print(f"Found {len(df_flagged)} flagged events. Sampling {len(df_sample)} to explain...")

    # --- (Optional) Step 3: Configure Gemini API ---
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("Warning: GEMINI_API_KEY environment variable not set. Using mocked explanations.", file=sys.stderr)
        client = None
    else:
        client = genai.Client()
        print("Gemini API client initialized.")
    # model = None # Force mocked path

    # --- Step 4: Generate explanations ---
    explanations = {}
    for _, row in df_sample.iterrows():
        event_id = row['event_id']
        
        if client:
            # This is where you would make the live API call
            event_data_json = row.to_json(indent=2)
            prompt = f"""
The following event was flagged by a machine learning model for potential GPS spoofing.
Please provide a brief, easy-to-understand explanation for a human reviewer based on the data.
Focus on the most likely reasons for the flag.

**Flagged Event Data:**
{event_data_json}
"""
            response = client.models.generate_content(model=MODEL_ID, contents=prompt)
            explanation = response.text
            # explanation = generate_mock_explanation(row) # Placeholder
        else:
            # Mocked path
            explanation = generate_mock_explanation(row)
        
        explanations[event_id] = explanation
        time.sleep(1)
    if client:
        client.close() 
    # --- Step 5: Prepare and save results ---
    # Create a list of dictionaries containing only the explained records
    records_to_save = []
    explained_ids = list(explanations.keys())
    
    # Filter the original predictions dataframe to get the scores/flags for our explained events
    df_explained_preds = df_sample[df_sample['event_id'].isin(explained_ids)]

    for _, row in df_explained_preds.iterrows():
        records_to_save.append({
            "event_id": row['event_id'],
            "spoof_score": row['spoof_score_ml'],
            "spoof_flag": int(row['spoof_flag_ml']),
            "explanation": explanations[row['event_id']]
        })

    # Save the processed records to the specified JSON file
    with open(results_path, 'w') as f:
        json.dump(records_to_save, f, indent=4)

    print(f"\nSuccessfully generated explanations for {len(records_to_save)} events.")
    print(f"Saved explained records to '{results_path}'.")

if __name__ == "__main__":
    main()
