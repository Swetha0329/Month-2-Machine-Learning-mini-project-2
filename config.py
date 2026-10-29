"""Central configuration for the PhishShift Message Risk Scanner."""

import os
from pathlib import Path

# Project root directory
BASE_DIR = Path(__file__).resolve().parent

# Main project directories
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SRC_DIR = BASE_DIR / "src"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"
TESTS_DIR = BASE_DIR / "tests"
REPORTS_DIR = BASE_DIR / "reports"
DOCS_DIR = BASE_DIR / "docs"

# Dataset configuration
DATASET_PATH = DATA_DIR / "phishshift_dataset.csv"

# Model configuration
MODEL_DIR = MODELS_DIR
MODEL_PATH = MODEL_DIR / "phishshift_model.joblib"
VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.joblib"
MODEL_METADATA_PATH = MODEL_DIR / "model_metadata.json"

# Application configuration
APP_NAME = "PhishShift"
APP_SUBTITLE = "Message Risk Scanner"
APP_HOST = "127.0.0.1"
APP_PORT = 5000
DEBUG_MODE = os.environ.get("PHISHSHIFT_DEBUG", "false").lower() == "true"

# Dataset validation
MINIMUM_DATASET_ROWS = 400

REQUIRED_DATASET_COLUMNS = [
    "message_id",
    "channel",
    "subject_or_title",
    "body_text",
    "sanitized_url_features",
    "sender_domain_type",
    "label",
    "campaign_group",
    "collected_date",
]

# Supported classification labels
LEGITIMATE_LABEL = "legitimate"
PHISHING_LABEL = "phishing"

ALLOWED_LABELS = {
    LEGITIMATE_LABEL,
    PHISHING_LABEL,
}

# Reproducibility
RANDOM_STATE = 42

# Upload limits and text-processing safeguards
MAX_CONTENT_LENGTH = 1 * 1024 * 1024
MAX_MESSAGE_LENGTH = 20_000

# Do not enable debug mode in a public deployment.
SECRET_KEY = os.environ.get("PHISHSHIFT_SECRET_KEY")

if not SECRET_KEY:
    SECRET_KEY = "local-development-only-change-before-deployment"