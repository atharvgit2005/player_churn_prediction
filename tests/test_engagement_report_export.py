from io import BytesIO

import numpy as np
from pypdf import PdfReader

from engagement_assistant import generate_engagement_optimization_report
from engagement_assistant.export import (
    MANDATORY_REPORT_SECTIONS,
    build_report_payload,
    generate_report_pdf_bytes,
    validate_report_payload,
)


def _sample_player_features():
    return {
        "PlayerID": "P-1042",
        "SessionsPerWeek": 2,
        "AvgSessionDurationMinutes": 8,
        "PlayTimeHours": 3,
        "InGamePurchases": 0,
        "PlayerLevel": 5,
        "AchievementsUnlocked": 2,
        "GameGenre": "Multiplayer RPG",
        "GameDifficulty": "Hard",
        "EngagementLevel": "Low",
    }


def test_report_schema_completeness():
    report, _ = generate_engagement_optimization_report(
        player_features=_sample_player_features(),
        churn_probability=0.82,
        player_identifier="P-1042",
    )

    payload = build_report_payload(report)

    assert set(MANDATORY_REPORT_SECTIONS).issubset(payload.keys())
    for section in MANDATORY_REPORT_SECTIONS:
        assert payload[section]
    assert validate_report_payload(payload) == []


def test_missing_noisy_data_behavior():
    noisy_player = {
        "PlayerID": "",
        "SessionsPerWeek": None,
        "AvgSessionDurationMinutes": "",
        "PlayTimeHours": np.nan,
        "InGamePurchases": 999999999,
        "PlayerLevel": -50,
        "AchievementsUnlocked": np.inf,
        "GameGenre": None,
        "GameDifficulty": "",
        "EngagementLevel": "Low",
    }

    report, _ = generate_engagement_optimization_report(
        player_features=noisy_player,
        churn_probability=0.91,
        player_identifier=None,
    )
    payload = build_report_payload(report)

    assert payload["title_block"]["player_id"] == "Unknown player"
    assert payload["player_behavior_summary"]
    assert report.data_quality_notes

    broken_payload = dict(payload)
    broken_payload["supporting_references"] = []
    missing_sections = validate_report_payload(broken_payload)
    assert "supporting_references" in missing_sections


def test_pdf_generation_success():
    report, _ = generate_engagement_optimization_report(
        player_features=_sample_player_features(),
        churn_probability=0.82,
        player_identifier="P-1042",
    )

    payload = build_report_payload(report)
    pdf_bytes = generate_report_pdf_bytes(payload)

    assert pdf_bytes.startswith(b"%PDF")
    assert len(pdf_bytes) > 0


def test_references_present_in_json_and_pdf():
    report, _ = generate_engagement_optimization_report(
        player_features=_sample_player_features(),
        churn_probability=0.82,
        player_identifier="P-1042",
    )

    report_json = report.to_dict()
    assert report_json["supporting_references"]

    payload = build_report_payload(report)
    assert payload["supporting_references"]

    pdf_bytes = generate_report_pdf_bytes(payload)
    pdf_reader = PdfReader(BytesIO(pdf_bytes))
    extracted_text = "\n".join(page.extract_text() or "" for page in pdf_reader.pages)

    first_reference = payload["supporting_references"][0]
    assert first_reference["title"] in extracted_text or first_reference["source"] in extracted_text
