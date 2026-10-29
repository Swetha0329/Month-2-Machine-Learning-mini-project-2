import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import app, analyze_message, find_risk_cues


@pytest.fixture
def client():
    app.config["TESTING"] = True
    with app.test_client() as test_client:
        yield test_client


def test_dashboard_opens(client):
    response = client.get("/")
    assert response.status_code == 200


def test_empty_message_is_handled(client):
    response = client.post("/", data={"message": ""})
    assert response.status_code == 200
    assert b"No message provided" in response.data


def test_risk_cues_detect_urgent_language():
    cues = find_risk_cues(
        "URGENT: Verify your password immediately."
    )
    assert any("Urgent" in cue for cue in cues)
    assert any("verification" in cue.lower() or "login" in cue.lower()
               or "account" in cue.lower() for cue in cues)


def test_risk_cues_detect_defanged_link():
    cues = find_risk_cues(
        "Review this link: hxxp://example[.]invalid"
    )
    assert any("link" in cue.lower() for cue in cues)


def test_analyze_message_returns_required_fields():
    result = analyze_message(
        "The team meeting starts at 10 AM in the project room."
    )

    required_fields = {
        "classification",
        "predicted_class",
        "decision_score",
        "risk_level",
        "risk_cues",
        "explanation",
        "is_uncertain",
    }

    assert required_fields.issubset(result.keys())
    assert isinstance(result["is_uncertain"], bool)


def test_blank_input_is_not_classified():
    result = analyze_message("   ")

    assert result["predicted_class"] == "Not analyzed"
    assert result["decision_score"] is None
    assert result["is_uncertain"] is True
