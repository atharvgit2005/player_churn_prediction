"""Agentic AI Game Engagement Optimization Assistant.

This package adds an explicit-state agent workflow that interprets churn risk
predictions + gameplay signals, retrieves retention strategies, and produces a
structured engagement report.
"""

from .engine import generate_engagement_optimization_report
from .export import build_report_payload, generate_report_pdf_bytes, validate_report_payload, validation_error_message
from .schema import EngagementOptimizationReport
from .retrieval import RetrievedStrategy
from .state import AgentState, AgentStep

__all__ = [
    "AgentState",
    "AgentStep",
    "EngagementOptimizationReport",
    "RetrievedStrategy",
    "build_report_payload",
    "generate_report_pdf_bytes",
    "generate_engagement_optimization_report",
    "validate_report_payload",
    "validation_error_message",
]
