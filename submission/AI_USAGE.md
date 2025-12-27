# AI Usage in the project

Tools used: Gemini Code Assist + Web Gemini
Model used: Gemini 2.5 Pro, Gemini 2.5 Flash (for testing out AI explanations)


## Prompts

### Prompt: Provide context of the project
```
For this project, we will need to design/synthesise a dataset for transaction-like events meant to be received by an Android SDK, then do location spoofing detection. Don't do anything yet, I will provide you the tasks step by step
```

### Prompt: Build data

```
First propose a data schema, and document it (including schema, range, taxonomy, distribution, limitation, etc.) in a new file submission/DATACARD.md. The data schema file should follow some following requrirements
- Must include event_id.
- Draft exactly 15 features, not including identifiers.
- Imagine what a modern Android SDK on most mid-range devices can possibly collect (while conforming to general platform policies/permissions)
- Include fields that are beyond basic lat/lon/timestamp, justify in the document.
- For each added field, please document how an SDK could obtain or infer it and why it should help in location spoofing detection, and expected range/value/type.
- Markdown file including the schema inside should be properly formated for easy preview.
```

Needs a bit of feedback loop. Initially there is only event_id, no identifier that ties back to a device, so Gemini suggested adding installation_id to trace journeys for a single device without relying on hardware ID, which are restricted for privacy reasons.


### Prompt: generate files
```
Based on the features in the data schema provided, create a file submission/src/generate_data.py. You don't need to run the file nor create the folder.
- The code should generate train.csv and test.csv in submission/data.
- train data should be labeled and test data should be unlabeled (but the ground truth should be known and saved).
- train should have around 10k rows and test should have around 3k rows.
- Spoof percentage should be within the range 10%-30% in train. The exact rate should be reported.
- Generated data should have realistic sensor noise and at least 5 types of common spoofing scenario.
```
Some problems:
- Had a problem with code execution in Gemini Code Assist. So I explicitly told the agent not to run the code.
- Minor problem with folder path. After pointing out, Gemini has fixed it.

Feedback loop discussion:

I asked why not include IP address, and Gemini responded 

> You are correct that comparing the location derived from an IP address with the GPS-reported location is a common technique used in fraud detection, especially for web traffic. However, I deliberately omitted an IP address field for these reasons:
> - Low Precision for Mobile: On mobile devices, IP geolocation is often highly inaccurate. A phone's public IP address belongs to the mobile carrier's network (e.g., Verizon, T-Mobile), and its reported location might be the carrier's central office, which can be tens or even hundreds of miles away from the user's actual physical location. Relying on this for fine-grained location validation would create a very high number of false positives.
> - SDK vs. Backend Data: An Android SDK can only see the device's local IP (e.g., 192.168.1.10), which is not useful for geolocation. The public IP address is typically captured on the backend server when the SDK sends its data. Therefore, a Geo/IP check is usually performed server-side by enriching the incoming data, rather than being a feature the SDK itself can generate.
> - Privacy: A user's public IP address is considered personal data, and its collection and use are subject to privacy regulations. While the other sensors are for on-device analysis of the physical environment, IP address is more about network identity.


### Prompt: Create rules based baselines 
```
Based on the data schema and generation script, create a baseline for the spoof detection task using rules/heuristics. 
- Only use the features from the generated data.
- Use at least 5 rules. 
- Example rules can include, but not limited to, teleportation and IP/Geo mismatch. 
- The baseline is meant to be imported and used later, do not generate any prediction yet.
- Save the baseline to the file submission/src/rules_baseline.py.
```

### Prompt: ML model
```
Create a file model_train_eval.py in submission/src. The file will do the following
- Load train and test data
- Train a RandomForestClassifier. You can create utility functions for proper feature engineering of the features in this same file. 
- Justify threshold using PR analysis, use the highest F1 threshold.
- Output the predictions of the test set as results.json. The JSON should have the following keys: event_id, spoof_score_ml, spoof_flag_ml.
- Also save the predictions of the test using the rules baseline approach, ML approach and hybrid approach to csvs for later analysis.
```

Problems:
- Some import error and key error, needed several chats for Gemini to fix.


### Prompt: Explainer
```
Create a new file submission/src/ai_helper.py which will do explainer to our model. The file can be run with argument `frac` which indicates how much percentage of data will be sampled to explain. Only create the file, do not run or install any dependencies yourself.
- Load the prediction results (ml_prediction.csv)
- Sample `frac` percentage of the data points with spoof_flag_ml=1
- Mock a prompt that can produce a natural language explanation for each data point.
- Call gemini-pro-2.5 programatically with the prompt using API key set in environment variable.
- Save the explanation to results.json with the following keys: event_id, spoof_score_ml, spoof_flag_ml, explanation. 
```

Feedback loop:
- The SDK code generated by Gemini was outdated, so I manually fixed it
- I used Gemini Flash 2.5 to test out some records (I don't have access to paid API) and save the results in results_sample.json

### Prompt: design.md
```
Create a design.md file that explain the flow of the spoof detection project, from ingestion→features→rules→ML→AI→decision. The file should not span more than 1 A4 page.
```

### Prompt: README.md and requirements.txt 
```
Create a requirements.txt file and a README.md file to indicate the run steps.
```