"""Agentic AI Game Engagement Optimization Assistant.

This package adds an explicit-state agent workflow that interprets churn risk
predictions + gameplay signals, retrieves retention strategies, and produces a
structured engagement report.
"""

from .engine import generate_engagement_optimization_report
from .schema import EngagementOptimizationReport
from .retrieval import RetrievedStrategy
from .state import AgentState, AgentStep

__all__ = [
    "AgentState",
    "AgentStep",
    "EngagementOptimizationReport",
    "RetrievedStrategy",
    "generate_engagement_optimization_report",
]
