from pathlib import Path
from datetime import datetime
from urllib.parse import urlsplit
import json
import logging
import os
import re

import joblib
from flask import (
    Flask,
    render_template,
    request,
    redirect,
    url_for,
    flash,
)

# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent

MODEL_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

MODEL_PATH = MODEL_DIR / "linear_svm.joblib"
METRICS_PATH = REPORTS_DIR / "model_metrics.json"
HISTORY_PATH = ROOT / "data" / "scan_history.json"

# --------------------------------------------------
# FLASK CONFIGURATION
# --------------------------------------------------

app = Flask(__name__)
app.secret_key = os.environ.get(
    "FLASK_SECRET_KEY",
    "phishshift-development-key"
)

app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024

logging.basicConfig(level=logging.INFO)

# --------------------------------------------------
# MODEL LOADING
# --------------------------------------------------

model = None
vectorizer = None
model_error = None

try:
    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        app.logger.info("Model loaded successfully: %s", MODEL_PATH)
    else:
        model_error = (
            "Model file not found. Expected: "
            "models/linear_svm.joblib"
        )
        app.logger.error(model_error)

except Exception as exc:
    model_error = f"Could not load the model: {exc}"
    app.logger.exception("Model loading failed")


def load_optional_vectorizer():
    """
    If the saved model is only a classifier (not a Pipeline),
    look for a separately saved vectorizer in the models folder.
    """
    if not MODEL_DIR.exists():
        return None

    for path in sorted(MODEL_DIR.glob("*.joblib")):
        name = path.name.lower()

        if path.resolve() == MODEL_PATH.resolve():
            continue

        if "vectorizer" in name or "tfidf" in name or "vectoriser" in name:
            try:
                loaded = joblib.load(path)
                app.logger.info("Vectorizer loaded: %s", path)
                return loaded
            except Exception:
                app.logger.exception("Could not load vectorizer %s", path)

    return None


if model is not None:
    vectorizer = load_optional_vectorizer()


def predict_label(message):
    """Predict from raw text, using a separate vectorizer if needed."""
    try:
        return model.predict([message])[0]

    except Exception:
        if vectorizer is None:
            raise

        features = vectorizer.transform([message])
        return model.predict(features)[0]


# --------------------------------------------------
# LABEL MAPPING
# --------------------------------------------------

PHISHING_LABELS = {
    "phishing",
    "phish",
    "spam",
    "malicious",
    "1",
    "true",
    "yes",
}

LEGITIMATE_LABELS = {
    "legitimate",
    "legit",
    "ham",
    "benign",
    "safe",
    "0",
    "false",
    "no",
}


def classify_label(raw_label):
    """Convert a model label into a readable category."""

    label = str(raw_label).strip().lower()

    if label in PHISHING_LABELS:
        return "Phishing"

    if label in LEGITIMATE_LABELS:
        return "Legitimate"

    if any(word in label for word in ("phishing", "malicious", "spam")):
        return "Phishing"

    if any(word in label for word in ("legitimate", "benign", "ham")):
        return "Legitimate"

    return "Needs Review"


def level_from_score(score, prediction=None):
    """Convert a risk score into a risk level label."""
    if prediction == "Needs Review":
        return "Review"

    try:
        score = float(score)
    except (TypeError, ValueError):
        return "Low"

    if score >= 70:
        return "High"

    if score >= 40:
        return "Medium"

    return "Low"


# --------------------------------------------------
# JSON FILE HELPERS
# --------------------------------------------------

def read_json(path, default):
    try:
        if path.exists():
            with path.open("r", encoding="utf-8") as file:
                return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        app.logger.warning("Could not read %s: %s", path, exc)

    return default


def save_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, indent=2, ensure_ascii=False)


def get_history():
    history = read_json(HISTORY_PATH, [])

    if not isinstance(history, list):
        return []

    return [item for item in history if isinstance(item, dict)]


# --------------------------------------------------
# MODEL METRICS (flexible loader)
# --------------------------------------------------

def _norm(key):
    return re.sub(r"[^a-z0-9]", "", str(key).lower())


def _to_float(value):
    if isinstance(value, bool):
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, str):
        text = value.strip()
        try:
            if text.endswith("%"):
                return float(text[:-1]) / 100
            return float(text)
        except ValueError:
            return None

    return None


def _iter_dicts(obj):
    queue = [obj]

    while queue:
        current = queue.pop(0)

        if isinstance(current, dict):
            yield current
            queue.extend(current.values())

        elif isinstance(current, list):
            queue.extend(current)


AVERAGE_KEYS = {"weightedavg", "macroavg", "weighted", "macro", "average", "overall"}


def find_metric(data, aliases):
    """
    Find a metric anywhere in a (possibly nested) JSON structure.
    Priority: top level, then weighted/macro averages, then anywhere.
    """
    wanted = {_norm(alias) for alias in aliases}

    def search(container):
        if not isinstance(container, dict):
            return None

        for key, value in container.items():
            if _norm(key) in wanted:
                number = _to_float(value)
                if number is not None:
                    return number

        return None

    # 1) top level
    found = search(data)
    if found is not None:
        return found

    all_dicts = list(_iter_dicts(data))

    # 2) average sections such as "weighted avg"
    for container in all_dicts:
        for key, value in container.items():
            if _norm(key) in AVERAGE_KEYS and isinstance(value, dict):
                found = search(value)
                if found is not None:
                    return found

    # 3) anywhere else
    for container in all_dicts:
        found = search(container)
        if found is not None:
            return found

    return None


def extract_metrics(data):
    return {
        "accuracy": find_metric(data, ["accuracy", "acc", "test_accuracy"]),
        "precision": find_metric(
            data, ["precision", "precision_score", "weighted_precision"]
        ),
        "recall": find_metric(
            data, ["recall", "recall_score", "weighted_recall"]
        ),
        "f1_score": find_metric(
            data, ["f1_score", "f1", "f1-score", "f_score", "weighted_f1"]
        ),
    }


def get_metrics():
    """
    Returns (metrics_dict, source_path_or_None).
    Looks at reports/model_metrics.json first, then other JSON
    files in reports/ and models/.
    """
    candidates = [METRICS_PATH]

    for folder in (REPORTS_DIR, MODEL_DIR):
        if folder.exists():
            for path in sorted(folder.glob("*.json")):
                if path not in candidates:
                    candidates.append(path)

    for path in candidates:
        if not path.exists():
            continue

        data = read_json(path, None)

        if not isinstance(data, (dict, list)):
            continue

        metrics = extract_metrics(data)

        if any(value is not None for value in metrics.values()):
            return metrics, str(path.relative_to(ROOT))

    empty = {
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1_score": None,
    }

    return empty, None


# --------------------------------------------------
# DASHBOARD STATISTICS
# --------------------------------------------------

def get_stats(history):
    phishing = 0
    legitimate = 0
    review = 0
    high = 0
    medium = 0
    low = 0

    for item in history:
        prediction = item.get("prediction")

        if prediction == "Phishing":
            phishing += 1
        elif prediction == "Legitimate":
            legitimate += 1
        else:
            review += 1

        level = str(
            item.get("risk_level")
            or level_from_score(item.get("risk_score"), prediction)
        ).lower()

        if level == "high":
            high += 1
        elif level == "medium":
            medium += 1
        elif level == "low":
            low += 1

    total = len(history)

    return {
        "total": total,
        "total_scans": total,
        "phishing": phishing,
        "phishing_count": phishing,
        "legitimate": legitimate,
        "legitimate_count": legitimate,
        "review": review,
        "uncertain": review,
        "uncertain_count": review,
        "high": high,
        "medium": medium,
        "low": low,
    }


def prepare_history_for_display(history):
    """Make sure every history item has the fields the page shows."""
    prepared = []

    for item in history:
        prediction = item.get("prediction", "Unknown")

        prepared.append({
            "message": item.get("preview", "Message unavailable"),
            "classification": prediction,
            "risk_score": item.get("risk_score"),
            "risk_level": (
                item.get("risk_level")
                or level_from_score(item.get("risk_score"), prediction)
            ),
            "timestamp": item.get("time", ""),
        })

    return prepared


# --------------------------------------------------
# URL ANALYSIS
# These checks do not visit or open any URL.
# --------------------------------------------------

URL_PATTERN = re.compile(
    r"""(?:https?://|www\.)[^\s<>"']+""",
    re.IGNORECASE,
)


def extract_urls(message):
    return URL_PATTERN.findall(message)


def analyze_urls(message):
    results = []

    for raw_url in extract_urls(message):
        cleaned = raw_url.rstrip(".,!?;:)]}")

        candidate = (
            cleaned
            if "://" in cleaned
            else "http://" + cleaned
        )

        try:
            parsed = urlsplit(candidate)
            hostname = (parsed.hostname or "").lower()
        except ValueError:
            hostname = ""

        indicators = []

        if not hostname:
            indicators.append("Could not parse the URL")
        else:
            if hostname.startswith("xn--") or ".xn--" in hostname:
                indicators.append("Internationalized domain name")

            if re.fullmatch(
                r"\d{1,3}(?:\.\d{1,3}){3}",
                hostname,
            ):
                indicators.append("IP address used as the host")

            if hostname.count(".") >= 3:
                indicators.append("Multiple subdomain levels")

            if "@" in cleaned:
                indicators.append("Contains an @ symbol")

        results.append({
            "url": cleaned,
            "host": hostname or "Unknown host",
            "indicators": (
                indicators
                if indicators
                else ["No simple URL warning detected"]
            ),
        })

    return results


# --------------------------------------------------
# TEXT-BASED RISK INDICATORS
# These are warning cues, not proof of phishing.
# --------------------------------------------------

def detect_risk_cues(message):
    cues = []

    patterns = [
        (
            r"\b(urgent|immediately|act now|final warning)\b",
            "Urgent or pressure-based wording",
        ),
        (
            r"\b(password|verify your account|login details|credentials)\b",
            "Requests account details or verification",
        ),
        (
            r"\b(prize|winner|claim your reward|free gift)\b",
            "Prize or reward wording",
        ),
        (
            r"\b(bank account|credit card|payment details|wire transfer)\b",
            "Financial information or payment wording",
        ),
        (
            r"\b(suspended|locked|unusual activity)\b",
            "Account threat or security warning",
        ),
    ]

    lower_message = message.lower()

    for pattern, description in patterns:
        if re.search(pattern, lower_message):
            cues.append(description)

    if extract_urls(message):
        cues.append("Contains one or more links")

    return cues


# --------------------------------------------------
# MESSAGE SCANNER
# --------------------------------------------------

def scan_message(message):
    if model is None:
        raise RuntimeError(
            model_error
            or "The trained model is unavailable."
        )

    raw_prediction = predict_label(message)

    prediction = classify_label(raw_prediction)

    cues = detect_risk_cues(message)
    urls = analyze_urls(message)

    # Heuristic score only. This is NOT model confidence
    # and must not be presented as a probability.
    risk_score = 15
    risk_score += min(len(cues) * 12, 48)
    risk_score += min(len(urls) * 10, 20)

    if prediction == "Phishing":
        risk_score += 25
    elif prediction == "Legitimate":
        risk_score -= 10

    risk_score = max(0, min(100, risk_score))

    risk_level = level_from_score(risk_score, prediction)

    explanations = {
        "Phishing": (
            "The machine-learning model classified this message "
            "as phishing. Avoid clicking its links or sharing "
            "sensitive information."
        ),
        "Legitimate": (
            "The machine-learning model classified this message "
            "as legitimate. This prediction does not guarantee "
            "that the message is safe."
        ),
        "Needs Review": (
            "The model returned a label that could not be mapped "
            "to a known category. Review the message manually."
        ),
    }

    return {
        "classification": prediction,
        "predicted_class": prediction,
        "is_uncertain": prediction == "Needs Review",
        "risk_level": risk_level,
        "risk_score": risk_score,
        "risk_cues": cues,
        "urls": urls,
        "explanation": explanations[prediction],
    }


# --------------------------------------------------
# MAIN PAGE
# --------------------------------------------------

@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    message = ""

    if request.method == "POST":
        message = request.form.get("message", "").strip()

        if not message:
            flash("Please enter a message to scan.", "error")

        elif len(message) > 20000:
            flash(
                "Please keep the message under 20,000 characters.",
                "error",
            )

        else:
            try:
                result = scan_message(message)

                history = get_history()

                history.insert(0, {
                    "time": datetime.now().strftime(
                        "%Y-%m-%d %H:%M:%S"
                    ),
                    "prediction": result["predicted_class"],
                    "risk_level": result["risk_level"],
                    "risk_score": result["risk_score"],
                    "preview": message[:140],
                })

                save_json(HISTORY_PATH, history[:100])

            except Exception as exc:
                app.logger.exception("Message scan failed")

                flash(
                    f"The message could not be scanned: {exc}",
                    "error",
                )

    history = get_history()
    metrics, metrics_source = get_metrics()

    return render_template(
        "index.html",
        result=result,
        message=message,
        history=prepare_history_for_display(history[:10]),
        stats=get_stats(history),
        metrics=metrics,
        metrics_source=metrics_source,
        model_available=model is not None,
    )


# --------------------------------------------------
# CLEAR SCAN HISTORY
# --------------------------------------------------

@app.route("/clear-history", methods=["POST"])
def clear_history():
    save_json(HISTORY_PATH, [])
    flash("Scan history cleared.", "success")

    return redirect(url_for("index"))


# --------------------------------------------------
# START APPLICATION
# --------------------------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))

    app.run(
        host="127.0.0.1",
        port=port,
        debug=False,
    )