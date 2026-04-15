"""Agentic AI Game Engagement Optimization Assistant.

This package adds an explicit-state agent workflow that interprets churn risk
predictions + gameplay signals and produces a structured engagement report.
"""

from .engine import generate_engagement_optimization_report
from .schema import EngagementOptimizationReport
from .state import AgentState, AgentStep

__all__ = [
    "AgentState",
    "AgentStep",
    "EngagementOptimizationReport",
    "generate_engagement_optimization_report",
]

