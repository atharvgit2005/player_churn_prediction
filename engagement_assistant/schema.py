from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .retrieval import RetrievedStrategy


@dataclass(frozen=True)
class Reference:
    """A human-readable reference that supports a recommendation."""

    title: str
    source: str
    note: Optional[str] = None
    url: Optional[str] = None


@dataclass(frozen=True)
class Recommendation:
    title: str
    rationale: str
    expected_impact: str
    effort: str
    risk: str
    metrics_to_track: List[str] = field(default_factory=list)
    references: List[Reference] = field(default_factory=list)
    supporting_signals: List[str] = field(default_factory=list)
    action_steps: List[str] = field(default_factory=list)
    confidence: Optional[float] = None
    uncertainty_notes: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class EngagementOptimizationReport:
    generated_at_utc: str
    player_identifier: Optional[str]

    player_behavior_summary: Dict[str, Any]
    churn_risk_interpretation: Dict[str, Any]
    engagement_and_retention_recommendations: List[Recommendation]

    supporting_references: List[Reference]
    ethical_and_ux_disclaimers: List[str]
    data_quality_notes: List[str] = field(default_factory=list)
    analysis_summary: Dict[str, Any] = field(default_factory=dict)
    retrieved_strategies: List[RetrievedStrategy] = field(default_factory=list)
    workflow_mode: str = "retrieval"

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @staticmethod
    def now_utc_iso() -> str:
        return datetime.now(timezone.utc).replace(microsecond=0).isoformat()
