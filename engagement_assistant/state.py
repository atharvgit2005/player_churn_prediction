from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class AgentStep(str, Enum):
    INGEST = "INGEST"
    ANALYZE = "ANALYZE"
    RECOMMEND = "RECOMMEND"
    FINALIZE = "FINALIZE"


@dataclass(frozen=True)
class AgentEvent:
    at_utc: str
    step: AgentStep
    message: str
    details: Optional[Dict[str, Any]] = None


@dataclass
class AgentState:
    """Explicit state container for the assistant workflow."""

    step: AgentStep = AgentStep.INGEST
    events: List[AgentEvent] = field(default_factory=list)

    # Inputs
    churn_probability: Optional[float] = None
    risk_level: Optional[str] = None
    player_identifier: Optional[str] = None
    player_features: Dict[str, Any] = field(default_factory=dict)

    # Derived / intermediate
    engagement_profile: Dict[str, Any] = field(default_factory=dict)
    data_quality_notes: List[str] = field(default_factory=list)

    def log(self, message: str, details: Optional[Dict[str, Any]] = None) -> None:
        self.events.append(
            AgentEvent(
                at_utc=datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
                step=self.step,
                message=message,
                details=details,
            )
        )

