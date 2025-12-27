# AI Spoofing Detection Project

This project contains a machine learning pipeline to detect GPS spoofing events. It includes data generation, model training, a rules-based baseline, and an AI-powered explanation helper.

## Project Structure

- `data/`: Contains the generated training and testing datasets.
- `src/`: Contains all the Python source code.
- `eval_report.ipynb`: A Jupyter Notebook for analysis and evaluation.
- `*.csv`: Prediction results from the different models.
- `*.json`: Raw and sampled output files.

## Setup

### Prerequisites

- Python 3.8+
- `pip` for package installation

### Dependencies

Install the required Python packages by running:
```bash
pip install -r requirements.txt
```

### API Key (Optional)

The `ai_helper.py` script can use the Gemini API to generate natural language explanations. To enable this, you must set your API key as an environment variable:

```bash
export GEMINI_API_KEY="YOUR_API_KEY"
```

If the `GEMINI_API_KEY` is not set, the script will fall back to using mocked, deterministic explanations.

## How to Run

The project should be run from the root directory (`ai-spoofing-detection/`). The scripts are located in the `submission/src/` directory.

### Step 1: Generate Data

This script creates the synthetic `train.csv` and `test.csv` datasets used for the project. The data will be saved in the `submission/data/` directory.

```bash
python submission/src/generate_data.py
```

### Step 2: Train Model and Generate Predictions

This script performs several key actions:
1.  Trains a Random Forest classifier on the training data.
2.  Finds an optimal classification threshold based on the F1-score.
3.  Applies the trained model to the test set to generate `ml_predictions.csv`.
4.  Applies the heuristic rules from `rules_baseline.py` to generate `rules_predictions.csv`.
5.  Creates a simple hybrid model and generates `hybrid_predictions.csv`.

```bash
python submission/src/model_train_eval.py
```

### Step 3: Generate AI Explanations

This script takes a sample of the events flagged by the ML model and generates natural language explanations for why they might be spoofed.

The `--frac` argument specifies the fraction of flagged events to sample (default is `0.1` or 10%).

```bash
# Explain 10% of flagged events (default)
python submission/src/ai_helper.py

# Explain 50% of flagged events
python submission/src/ai_helper.py --frac 0.5
```

The output will be saved to `submission/results_sample.json`.

### Step 4: Review the Evaluation

The `submission/eval_report.ipynb` notebook is provided for analyzing the results. You can use Jupyter Lab or Jupyter Notebook to open it and run the cells to see the performance metrics, visualizations (like the PR curve), and error analysis.

```bash
# Make sure you are in the ai-spoofing-detection directory
jupyter notebook submission/eval_report.ipynb
```
