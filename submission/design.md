# Spoof Detection System Design

This document outlines the architecture of the AI spoofing detection project, detailing the flow from data ingestion to the final decision-making process.

## System Workflow

The system is designed as a multi-stage pipeline that processes location events to identify and flag potential spoofing activities. The core flow is as follows:

`Ingestion → Feature Engineering → Parallel Detection (Rules & ML) → AI Explanation → Final Decision`

![System Flow](https://i.imgur.com/3T1fB1N.png) 
*A high-level diagram of the detection pipeline.*

---

### 1. Ingestion (`generate_data.py`)

The pipeline begins with the ingestion of raw event data. For this project, a synthetic dataset is generated to simulate real-world scenarios.

-   **Input:** Basic device sensor readings (latitude, longitude, timestamp, altitude, speed, etc.).
-   **Process:** The `generate_data.py` script creates two types of journeys:
    1.  **Normal:** Simulates users who are stationary, walking, or driving.
    2.  **Spoofed:** Simulates common attack vectors, including:
        -   **Mock Provider:** Using developer tools to fake location.
        -   **Teleport:** Instantaneous, physically impossible jumps in location.
        -   **Bot Replay:** Perfectly repeated and consistent sensor data.
        -   **Frozen Location:** GPS coordinates remain static while other sensors indicate movement.
        -   **Sensor Mismatch:** Inconsistent readings between different sensors (e.g., altitude and pressure).
-   **Output:** Raw event logs with a ground truth `spoofed` label, split into `train.csv` and `test.csv`.

### 2. Feature Engineering (`model_train_eval.py`)

Once the raw data is ingested, it is enriched with additional features to improve model performance.

-   **Process:** The `feature_engineer` function creates new features from the raw inputs:
    -   **Time-based:** Extracts `hour` and `day_of_week` from the timestamp.
    -   **Interaction:** Creates features that capture relationships between sensors, such as `speed * pressure_hpa`.
    -   **Categorical Encoding:** Converts `wifi_bssid` and `cell_tower_id` into a numerical format using one-hot encoding.
-   **Output:** A feature-rich dataset ready for model consumption.

### 3. Parallel Detection (Rules & ML)

The core of the detection logic runs two models in parallel: a simple, fast rules-based baseline and a more sophisticated machine learning model.

#### 3.1. Rules-Based Model (`rules_baseline.py`)

This model uses a set of hardcoded heuristics to catch obvious spoofing patterns.

-   **Process:** Applies five key rules:
    1.  **Mock Location Flag:** Checks if the OS-level mock location setting is enabled.
    2.  **Impossible Speed:** Flags events that imply travel faster than the speed of sound.
    3.  **Perfect Accuracy:** Flags events with unnaturally perfect sensor readings (e.g., `horizontal_accuracy = 1.0`).
    4.  **Frozen Location:** Detects if coordinates are static while the device reports non-zero speed.
    5.  **Altitude-Pressure Mismatch:** Flags inconsistencies between barometric pressure and altitude readings.
-   **Output:** A binary prediction (`1` for spoof, `0` for normal). This model provides high recall but low precision.

#### 3.2. Machine Learning Model (`model_train_eval.py`)

This model uses a `RandomForestClassifier` to learn complex, non-linear patterns from the full feature set.

-   **Process:**
    1.  A Random Forest model is trained on the engineered feature set.
    2.  A threshold is optimized on a validation set to balance precision and recall, maximizing the F1 score.
    3.  The model predicts a `spoof_score_ml` (a probability from 0 to 1).
-   **Output:** A probabilistic score and a binary flag. This model provides high precision.

### 4. AI-Powered Explanation (`ai_helper.py`)

To improve transparency and aid human review, an AI-powered explanation module is included.

-   **Process:**
    1.  The system samples events that were flagged as spoofed by the ML model.
    2.  For each flagged event, it prepares a prompt containing the event's full feature set.
    3.  It calls a Large Language Model (Gemini Pro) to generate a concise, human-readable explanation for why the event was deemed suspicious.
-   **Output:** A JSON file (`results_sample.json`) containing the event ID, ML score, and a natural language `explanation`.

### 5. Final Decision Logic

The final stage consolidates the outputs and makes a decision. The `eval_report.ipynb` notebook is used to analyze the performance of three different strategies:

1.  **Rules-Only:** Use only the output of the rules-based model. (High Recall, Low Precision)
2.  **ML-Only:** Use only the output of the machine learning model. (High Precision, Moderate Recall)
3.  **Hybrid:** Combines the scores from both models. The current implementation uses a simple 50/50 average, which defaults to the rules-based output.

**Conclusion:** The evaluation shows that the **ML-only model provides the best balance of precision and recall (F1 Score: 0.77)**. For a production system, this model would be the primary choice. Future work could involve creating a more sophisticated hybrid model, such as using the rules output as a feature for the ML model, to further improve performance.
