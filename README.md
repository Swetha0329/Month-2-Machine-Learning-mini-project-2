# PhishShift — Message Risk Scanner

## Overview
PhishShift is a machine-learning web application that classifies messages as legitimate or phishing and highlights potential risk cues. It is a decision-support tool, not a guarantee of safety.

## Features
- Flask web dashboard
- TF-IDF text processing
- Linear SVM and Logistic Regression models
- Rule-based baseline
- Risk-cue explanations and uncertain-message review
- Dataset preparation, deduplication analysis, and EDA
- Saved model comparison and acceptance-test evidence

## Technology
Python, Flask, scikit-learn, pandas, NumPy, Matplotlib, TF-IDF, Linear SVM, Logistic Regression.

## Run locally
1. Activate your virtual environment.
2. Install dependencies: `pip install -r requirements.txt`
3. Start the app: `py app.py`
4. Open `http://127.0.0.1:5000`

## Model evaluation
The derived similarity-group holdout evaluation reported:
- Linear SVM: accuracy 98.77%, precision 98.71%, recall 98.15%, F1 98.43%, false-positive rate 0.83%.
- Logistic Regression: accuracy 98.38%, precision 97.74%, recall 98.15%, F1 97.94%, false-positive rate 1.48%.

These results use text-similarity groups, not verified campaign IDs. They should not be described as genuine campaign-separated performance.

## Acceptance tests
Evidence: `reports/acceptance_tests/acceptance_test_evidence.csv`.
Fourteen preliminary synthetic tests ran; twelve matched expected labels and two need review. This small test set is not a complete evaluation.

## Dataset and limitations
The dataset was obtained from an external source and prepared for this project. Verify and document the exact source license and provenance before submission. The internship problem statement did not provide a dataset. Mentor-provided unseen campaign data is still required for the specified campaign-separated evaluation.

Suspicious URLs are examples only. Do not open them. The model may produce false positives and miss phishing messages.

## Main artifacts
- `data/`: datasets and preparation outputs
- `models/`: saved machine-learning pipelines
- `reports/eda/`: exploratory data analysis charts and summary
- `reports/metrics/`: model and baseline evaluation
- `reports/acceptance_tests/`: preliminary acceptance-test evidence
- `app.py`: Flask application
- `templates/index.html`: dashboard interface
- `requirements.txt`: Python dependencies
