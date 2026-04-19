from __future__ import annotations

from io import BytesIO
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import ListFlowable, ListItem, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .schema import EngagementOptimizationReport

MANDATORY_REPORT_SECTIONS = [
    "title_block",
    "player_behavior_summary",
    "churn_risk_interpretation",
    "engagement_and_retention_recommendations",
    "supporting_references",
    "ethical_and_ux_disclaimers",
]


def _clean_text(value: Any, fallback: str = "Not available") -> str:
    text = " ".join(str(value).split()) if value is not None else ""
    return text if text else fallback


def _stringify_mapping(mapping: Mapping[str, Any], prefix: str = "") -> List[str]:
    lines: List[str] = []
    for key, value in mapping.items():
        label = key.replace("_", " ").title()
        if isinstance(value, Mapping):
            nested_prefix = f"{prefix}{label}: " if not prefix else f"{prefix}{label} > "
            lines.extend(_stringify_mapping(value, prefix=nested_prefix))
        elif isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            rendered = ", ".join(_clean_text(item, fallback="Unknown") for item in value) or "Not available"
            lines.append(f"{prefix}{label}: {rendered}")
        else:
            lines.append(f"{prefix}{label}: {_clean_text(value)}")
    return lines


def build_report_payload(report: EngagementOptimizationReport) -> Dict[str, Any]:
    recommendations: List[Dict[str, Any]] = []
    for recommendation in report.engagement_and_retention_recommendations:
        recommendations.append(
            {
                "title": _clean_text(recommendation.title),
                "rationale": _clean_text(recommendation.rationale),
                "expected_impact": _clean_text(recommendation.expected_impact),
                "effort": _clean_text(recommendation.effort),
                "risk": _clean_text(recommendation.risk),
                "action_steps": list(recommendation.action_steps) or ["No action steps provided."],
                "metrics_to_track": list(recommendation.metrics_to_track) or ["No metrics specified."],
                "supporting_signals": list(recommendation.supporting_signals) or ["No supporting signals provided."],
                "references": [
                    {
                        "title": _clean_text(reference.title),
                        "source": _clean_text(reference.source),
                        "note": _clean_text(reference.note),
                        "url": _clean_text(reference.url),
                    }
                    for reference in recommendation.references
                ],
            }
        )

    supporting_references = [
        {
            "title": _clean_text(reference.title),
            "source": _clean_text(reference.source),
            "note": _clean_text(reference.note),
            "url": _clean_text(reference.url),
        }
        for reference in report.supporting_references
    ]

    return {
        "title_block": {
            "title": "Engagement Optimization Report",
            "player_id": _clean_text(report.player_identifier, fallback="Unknown player"),
            "timestamp": _clean_text(report.generated_at_utc),
        },
        "player_behavior_summary": _stringify_mapping(report.player_behavior_summary),
        "churn_risk_interpretation": _stringify_mapping(report.churn_risk_interpretation),
        "engagement_and_retention_recommendations": recommendations,
        "supporting_references": supporting_references,
        "ethical_and_ux_disclaimers": [
            _clean_text(disclaimer) for disclaimer in report.ethical_and_ux_disclaimers if _clean_text(disclaimer)
        ],
    }


def validate_report_payload(payload: Mapping[str, Any]) -> List[str]:
    missing_sections: List[str] = []
    for section in MANDATORY_REPORT_SECTIONS:
        value = payload.get(section)
        if value is None:
            missing_sections.append(section)
            continue
        if isinstance(value, str) and not value.strip():
            missing_sections.append(section)
            continue
        if isinstance(value, Mapping) and not any(_clean_text(item) for item in value.values()):
            missing_sections.append(section)
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if not value:
                missing_sections.append(section)
                continue
            if not any(_clean_text(item) for item in value):
                missing_sections.append(section)
                continue
    return missing_sections


def validation_error_message(missing_sections: Iterable[str]) -> str:
    missing = [section.replace("_", " ").title() for section in missing_sections]
    return "Cannot export PDF. Missing required sections: " + ", ".join(missing)


def generate_report_pdf_bytes(payload: Mapping[str, Any]) -> bytes:
    missing_sections = validate_report_payload(payload)
    if missing_sections:
        raise ValueError(validation_error_message(missing_sections))

    buffer = BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=LETTER,
        rightMargin=0.6 * inch,
        leftMargin=0.6 * inch,
        topMargin=0.6 * inch,
        bottomMargin=0.6 * inch,
    )
    styles = getSampleStyleSheet()
    title_style = styles["Title"]
    section_style = ParagraphStyle(
        "SectionHeader",
        parent=styles["Heading2"],
        fontSize=13,
        textColor=colors.HexColor("#1d4ed8"),
        spaceBefore=8,
        spaceAfter=6,
    )
    body_style = styles["BodyText"]
    body_style.leading = 14

    story: List[Any] = []

    title_block = payload["title_block"]
    story.append(Paragraph(_clean_text(title_block.get("title")), title_style))
    story.append(Paragraph(f"Player ID: {_clean_text(title_block.get('player_id'))}", body_style))
    story.append(Paragraph(f"Generated: {_clean_text(title_block.get('timestamp'))}", body_style))
    story.append(Spacer(1, 0.18 * inch))

    story.append(Paragraph("Player Behavior Summary", section_style))
    for line in payload["player_behavior_summary"]:
        story.append(Paragraph(line, body_style))
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("Churn Risk Interpretation", section_style))
    for line in payload["churn_risk_interpretation"]:
        story.append(Paragraph(line, body_style))
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("Engagement & Retention Recommendations", section_style))
    table_rows = [["Recommendation", "Effort", "Risk", "Expected Impact"]]
    for recommendation in payload["engagement_and_retention_recommendations"]:
        table_rows.append(
            [
                _clean_text(recommendation.get("title")),
                _clean_text(recommendation.get("effort")),
                _clean_text(recommendation.get("risk")),
                _clean_text(recommendation.get("expected_impact")),
            ]
        )
    recommendation_table = Table(table_rows, colWidths=[2.1 * inch, 0.9 * inch, 0.8 * inch, 2.3 * inch])
    recommendation_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ]
        )
    )
    story.append(recommendation_table)
    story.append(Spacer(1, 0.12 * inch))

    bullet_items = []
    for recommendation in payload["engagement_and_retention_recommendations"]:
        action_steps = recommendation.get("action_steps", [])
        rationale = _clean_text(recommendation.get("rationale"))
        bullet_text = f"{_clean_text(recommendation.get('title'))}: {rationale}"
        if action_steps:
            bullet_text += f" Action steps: {'; '.join(_clean_text(step) for step in action_steps)}."
        bullet_items.append(ListItem(Paragraph(bullet_text, body_style)))
    story.append(ListFlowable(bullet_items, bulletType="bullet", start="circle"))
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("Supporting References", section_style))
    reference_items = []
    for reference in payload["supporting_references"]:
        note = _clean_text(reference.get("note"))
        url = _clean_text(reference.get("url"))
        reference_items.append(
            ListItem(
                Paragraph(
                    f"{_clean_text(reference.get('title'))} ({_clean_text(reference.get('source'))}). {note}. {url}",
                    body_style,
                )
            )
        )
    story.append(ListFlowable(reference_items, bulletType="bullet", start="square"))
    story.append(Spacer(1, 0.12 * inch))

    story.append(Paragraph("Ethical & User-Experience Disclaimers", section_style))
    disclaimer_items = [ListItem(Paragraph(_clean_text(item), body_style)) for item in payload["ethical_and_ux_disclaimers"]]
    story.append(ListFlowable(disclaimer_items, bulletType="bullet", start="square"))

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
