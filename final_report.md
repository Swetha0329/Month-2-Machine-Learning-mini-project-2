PHISHSHIFT — MESSAGE RISK SCANNER
DRAFT FINAL PROJECT REPORT

1. ABSTRACT
PhishShift is a machine-learning-based message risk scanner designed to help users identify potentially phishing messages. It uses TF-IDF text features and compares Linear Support Vector Classification with Logistic Regression, alongside a rule-based baseline. A Flask dashboard presents a classification and risk-related explanation. The project includes dataset preparation, duplicate and similarity analysis, exploratory data analysis, saved models, and preliminary acceptance tests. In the derived similarity-group holdout evaluation, Linear SVM achieved 98.77% accuracy and an F1 score of 98.43%; Logistic Regression achieved 98.38% accuracy and an F1 score of 97.94%. These results do not establish performance on verified unseen campaigns. Fourteen preliminary synthetic tests were run, with twelve matching their expected labels. A legitimate password-reset message was flagged as phishing, and a phishing message with a defanged URL was missed. These failures demonstrate the need for cautious interpretation. PhishShift is intended as decision support, not a guarantee of safety.

2. PROBLEM STATEMENT AND OBJECTIVES
Phishing messages can use urgency, account threats, requests for sensitive information, and deceptive links. The objective is to build a web tool that classifies message text, compares machine-learning approaches, explains possible risk cues, and handles uncertainty responsibly.

3. DATASET AND PROVENANCE
The internship problem statement specified a dataset schema but did not supply a dataset. The project uses an externally sourced email dataset mapped to legitimate and phishing labels. The prepared dataset contains 17,485 messages: 10,968 legitimate and 6,517 phishing. The exact source license and complete provenance must be verified and recorded before final submission. Dataset labels may not represent all real-world message types.

4. DATA PREPARATION AND EXPLORATORY ANALYSIS
Preparation outputs include a prepared CSV, template-group mapping, and similarity candidates. Three EDA figures are available: class distribution, message-length distribution, and top words by class. Similarity analysis found 5,600 candidate pairs and 3,573 messages in groups larger than one; the largest group contained 68 messages. These are text-similarity-derived groups, not verified campaign identities. The grouping method is approximate and may not capture all duplicates or campaign relationships.

5. MODELS AND EVALUATION
The saved pipelines use TF-IDF followed by LinearSVC or LogisticRegression. A derived similarity-group holdout used 13,909 training messages and 3,576 test messages, with zero overlapping derived groups.

Model | Accuracy | Precision | Recall | F1 | False-positive rate
Linear SVM | 0.9877 | 0.9871 | 0.9815 | 0.9843 | 0.0083
Logistic Regression | 0.9838 | 0.9774 | 0.9815 | 0.9794 | 0.0148

These metrics must be interpreted cautiously because the holdout is based on derived similarity groups rather than independently verified campaigns. The mentor-provided unseen synthetic campaign and legitimate business messages remain necessary to complete the required campaign-separated evaluation.

6. ACCEPTANCE TESTS AND ERRORS
Fourteen preliminary synthetic cases were recorded in reports/acceptance_tests/acceptance_test_evidence.csv. Twelve predictions matched the expected labels. AT01 incorrectly classified a legitimate password-reset notice as phishing. AT03a incorrectly classified a phishing message containing a defanged URL as legitimate. These examples show false-positive and missed-phishing risks. The set is preliminary and does not replace all required acceptance tests or calibrated uncertainty testing.

7. SAFETY, LIMITATIONS, AND ETHICS
The application is decision support only. Users should not treat predictions as proof that a message is safe or malicious. Suspicious URLs should not be opened or fetched. Real credentials, mailbox dumps, and personal data must not be used. The model may fail on unfamiliar wording, URL obfuscation, punctuation changes, new campaigns, and distribution drift. The abstention threshold requires further validation. Dataset licensing and source attribution must be verified before submission.

8. CONCLUSION AND FUTURE WORK
PhishShift has a working Flask interface, saved TF-IDF classification pipelines, EDA outputs, model-comparison results, and preliminary test evidence. Next steps are to verify dataset licensing and provenance, document deduplication before splitting, test on mentor-provided campaign-separated data, evaluate robustness and false positives on urgent legitimate messages, validate the abstention threshold, and complete the SRS, final report, README, and demonstration.

APPENDIX: EVIDENCE
- reports/metrics/fair_model_comparison.json
- reports/metrics/rules_baseline.json
- reports/eda/eda_summary.json
- reports/eda/01_class_distribution.png
- reports/eda/02_message_length_distribution.png
- reports/eda/03_top_words_by_class.png
- reports/acceptance_tests/acceptance_test_evidence.csv

This report is a draft. Add verified references, dataset license details, architecture screenshots, repository link, and the original SRS requirements before submission.

9. ADDITIONAL VERIFICATION EVIDENCE
Automated application tests are available in tests/test_app.py. The current test run completed six tests successfully. These tests cover dashboard availability, empty-input handling, risk-cue detection, classification result structure, and blank-input behavior. They do not establish overall phishing-detection accuracy or validate the abstention threshold.

A deduplication audit is available at reports/metrics/deduplication_audit.json. It records 17,522 source rows and 17,485 prepared rows, with zero repeated normalized_text values remaining in the prepared dataset. The difference in row counts should not be interpreted as duplicates alone because preprocessing may also remove unusable records.

Dataset integrity information is recorded in data/SOURCE_MANIFEST.md, including the SHA-256 checksum of the local source CSV. The original source revision and applicable licence or redistribution permissions remain unverified.

The campaign-group audit is available at reports/metrics/campaign_group_audit.json. Because the dataset does not provide verified campaign identifiers, the current evaluation cannot establish generalization to independently verified unseen campaigns.

The preliminary acceptance-test evidence remains at reports/acceptance_tests/acceptance_test_evidence.csv. It records 14 cases, including two failures: a legitimate password-reset message classified as phishing and a defanged-URL phishing message classified as legitimate.


10. REFERENCES

[1] Zefang Liu, "Phishing Email Dataset," Hugging Face Datasets.
    https://huggingface.co/datasets/zefang-liu/phishing-email-dataset
    Accessed: October 29, 2026.

[2] Scikit-learn Developers, " scikit-learn: Machine Learning in Python."
    https://scikit-learn.org/
    Accessed: October 29, 2026.

[3] Pallets Projects, "Flask Documentation."
    https://flask.palletsprojects.com/
    Accessed: October 29, 2026.

[4] Python Software Foundation, "Python Documentation."
    https://docs.python.org/3/
    Accessed: October 29, 2026.

Note: The dataset card lists a licence, but the licence and redistribution permissions applicable to the underlying original dataset have not yet been independently verified. These references do not imply that permission to redistribute the dataset has been established.

