import csv
import joblib
from pathlib import Path
from datetime import datetime

model = joblib.load("models/linear_svm.joblib")

tests = [
    ("AT01", "Legitimate password reset", "Your password reset request was received. If this was you, follow your company's normal process.", "legitimate"),
    ("AT02", "Unfamiliar phishing wording", "Your mailbox access is paused. Confirm your account immediately at http://example.invalid/verify", "phishing"),
    ("AT03a", "Defanged URL", "Review the shared document at hxxp://example[.]invalid/file", "phishing"),
    ("AT03b", "Punctuation variation", "REVIEW!!! shared-document... access now", "phishing"),
    ("CP01", "Harmless 1", "The team meeting starts at 10 AM in the project room.", "legitimate"),
    ("CP02", "Harmless 2", "Your assignment feedback is available on the college portal.", "legitimate"),
    ("CP03", "Harmless 3", "Reminder: submit your internship attendance by Friday.", "legitimate"),
    ("CP04", "Harmless 4", "The library will close at 5 PM today.", "legitimate"),
    ("CP05", "Harmless 5", "Please review the presentation before tomorrow's meeting.", "legitimate"),
    ("CP06", "Harmless 6", "Your lab session has been moved to Room 204.", "legitimate"),
    ("CP07", "Harmless 7", "The department seminar begins at 2 PM.", "legitimate"),
    ("CP08", "Harmless 8", "Thank you for registering for the campus workshop.", "legitimate"),
    ("ERR01", "Phishing example", "URGENT: Verify your password immediately at http://example.invalid/login", "phishing"),
    ("ERR02", "Legitimate example", "Your college project review is scheduled for tomorrow at 10 AM. Bring your report.", "legitimate"),
]

texts = [t[2] for t in tests]
predictions = model.predict(texts)

out = Path("reports/acceptance_tests")
out.mkdir(parents=True, exist_ok=True)
file = out / "acceptance_test_evidence.csv"

with file.open("w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["test_id", "test_name", "message", "expected_label", "predicted_label", "result", "test_time"])
    for test, pred in zip(tests, predictions):
        writer.writerow([test[0], test[1], test[2], test[3], pred,
                         "PASS" if pred == test[3] else "REVIEW",
                         datetime.now().isoformat(timespec="seconds")])

print("Saved:", file)
for test, pred in zip(tests, predictions):
    print(test[0], "| expected:", test[3], "| predicted:", pred,
          "|", "PASS" if pred == test[3] else "REVIEW")
print("\nNote: Synthetic preliminary tests only; campaign holdout and threshold behavior need separate validation.")
