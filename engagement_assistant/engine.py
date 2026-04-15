from __future__ import annotations

from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np

from .schema import EngagementOptimizationReport, Recommendation, Reference
from .state import AgentState, AgentStep


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        number = float(value)
        if np.isnan(number):
            return None
        if np.isinf(number):
            return None
        return number
    except Exception:
        return None


def _first_present(d: Mapping[str, Any], candidates: Iterable[str]) -> Tuple[Optional[str], Any]:
    lookup = {str(k).lower(): k for k in d.keys()}
    for name in candidates:
        key = lookup.get(name.lower())
        if key is not None:
            return str(key), d[key]
    return None, None


def _risk_bucket(probability: float) -> str:
    if probability <= 0.30:
        return "Low"
    if probability <= 0.70:
        return "Medium"
    return "High"


def _build_engagement_profile(state: AgentState) -> Dict[str, Any]:
    feats = state.player_features
    notes: List[str] = []

    # Common columns in the provided dataset (plus a few robust aliases).
    _, sessions = _first_present(feats, ["SessionsPerWeek", "sessions_per_week", "session_frequency"])
    _, avg_minutes = _first_present(feats, ["AvgSessionDurationMinutes", "avg_session_duration_minutes"])
    _, play_hours = _first_present(feats, ["PlayTimeHours", "play_time_hours"])
    _, purchases = _first_present(feats, ["InGamePurchases", "in_game_purchases"])
    _, level = _first_present(feats, ["PlayerLevel", "player_level", "Level"])
    _, achievements = _first_present(feats, ["AchievementsUnlocked", "achievements_unlocked"])
    _, genre = _first_present(feats, ["GameGenre", "game_genre", "Genre"])
    _, difficulty = _first_present(feats, ["GameDifficulty", "game_difficulty", "Difficulty"])
    _, engagement_level = _first_present(feats, ["EngagementLevel", "engagement_level"])

    sessions_f = _safe_float(sessions)
    avg_minutes_f = _safe_float(avg_minutes)
    play_hours_f = _safe_float(play_hours)
    purchases_f = _safe_float(purchases)
    level_f = _safe_float(level)
    achievements_f = _safe_float(achievements)

    if sessions is None:
        notes.append("Sessions per week is missing; session-frequency based recommendations are less certain.")
    if avg_minutes is None:
        notes.append("Average session duration is missing; session-length based recommendations are less certain.")
    if purchases is None:
        notes.append("In-game purchases is missing; monetization-friendly strategies are conservative by default.")

    engagement_score = None
    if sessions_f is not None and avg_minutes_f is not None:
        engagement_score = sessions_f * avg_minutes_f

    return {
        "sessions_per_week": sessions_f,
        "avg_session_duration_minutes": avg_minutes_f,
        "play_time_hours": play_hours_f,
        "in_game_purchases": purchases_f,
        "player_level": level_f,
        "achievements_unlocked": achievements_f,
        "game_genre": genre,
        "game_difficulty": difficulty,
        "engagement_level_observed": engagement_level,
        "engagement_score_proxy": engagement_score,
        "_data_quality_notes": notes,
    }


def _behavior_summary(profile: Mapping[str, Any]) -> Dict[str, Any]:
    # Keep it structured and robust to missing values.
    summary: Dict[str, Any] = {
        "engagement_overview": {},
        "progression_overview": {},
        "monetization_overview": {},
        "context": {},
        "interpretation_notes": [],
    }

    sessions = profile.get("sessions_per_week")
    avg_minutes = profile.get("avg_session_duration_minutes")
    score = profile.get("engagement_score_proxy")

    if sessions is not None:
        summary["engagement_overview"]["sessions_per_week"] = sessions
    else:
        summary["interpretation_notes"].append("Sessions/week unavailable.")

    if avg_minutes is not None:
        summary["engagement_overview"]["avg_session_duration_minutes"] = avg_minutes
    else:
        summary["interpretation_notes"].append("Avg session duration unavailable.")

    if score is not None:
        summary["engagement_overview"]["engagement_score_proxy"] = score

    level = profile.get("player_level")
    achievements = profile.get("achievements_unlocked")
    if level is not None:
        summary["progression_overview"]["player_level"] = level
    if achievements is not None:
        summary["progression_overview"]["achievements_unlocked"] = achievements

    purchases = profile.get("in_game_purchases")
    if purchases is not None:
        summary["monetization_overview"]["in_game_purchases"] = purchases

    summary["context"]["game_genre"] = profile.get("game_genre")
    summary["context"]["game_difficulty"] = profile.get("game_difficulty")
    summary["context"]["engagement_level_observed"] = profile.get("engagement_level_observed")

    return summary


def _base_references() -> List[Reference]:
    # Internal references (placeholders until Commit 2 adds retrieval-backed citations).
    return [
        Reference(
            title="Retention pattern: reduce early friction and increase short-term goals",
            source="Internal playbook (Commit 1 baseline)",
            note="General industry guidance; not game-specific yet.",
        ),
        Reference(
            title="Engagement loop: trigger → action → variable reward → investment",
            source="Internal playbook (Commit 1 baseline)",
            note="Used as a mental model; requires validation against your game telemetry.",
        ),
    ]


def _recommendations(profile: Mapping[str, Any], churn_probability: float, risk_level: str) -> List[Recommendation]:
    refs = _base_references()

    sessions = profile.get("sessions_per_week")
    avg_minutes = profile.get("avg_session_duration_minutes")
    purchases = profile.get("in_game_purchases")
    difficulty = str(profile.get("game_difficulty") or "").strip().lower()

    recs: List[Recommendation] = []

    # 1) Session frequency nudge
    if sessions is None or sessions < 3:
        recs.append(
            Recommendation(
                title="Bring the player back with a lightweight return hook",
                rationale="Low or unknown weekly session frequency correlates with churn risk; a short, low-friction goal can re-establish a habit.",
                expected_impact="Increase sessions/week and reduce time-to-next-session for at-risk players.",
                effort="Low",
                risk="Low (avoid spam; cap notifications and offer opt-out).",
                metrics_to_track=["sessions_per_week", "time_to_next_session", "D1/D7 retention"],
                references=[refs[0]],
            )
        )

    # 2) Session length / content pacing
    if avg_minutes is None or avg_minutes < 10:
        recs.append(
            Recommendation(
                title="Add a 5–8 minute 'quick win' loop",
                rationale="Short sessions may indicate the player isn't reaching rewarding moments quickly enough.",
                expected_impact="Improve session completion rate and perceived progress.",
                effort="Medium",
                risk="Medium (ensure rewards don't distort economy or progression).",
                metrics_to_track=["avg_session_duration_minutes", "mission_completion_rate", "session_dropoff_step"],
                references=[refs[1]],
            )
        )

    # 3) Difficulty smoothing (simple heuristic; extension can be upgraded later)
    if difficulty in {"hard", "high", "expert"} and risk_level in {"Medium", "High"}:
        recs.append(
            Recommendation(
                title="Offer optional difficulty relief without reducing autonomy",
                rationale="High difficulty combined with elevated churn risk can indicate frustration; optional assists preserve agency.",
                expected_impact="Reduce fail-streak exits and improve mid-session continuation.",
                effort="Medium",
                risk="Medium (avoid making skilled players feel patronized; keep it optional).",
                metrics_to_track=["fail_streak_length", "level_retry_rate", "quit_during_fail_streak"],
                references=[refs[0]],
            )
        )

    # 4) Monetization-safe re-engagement (only if purchases are low/unknown)
    if purchases is None or purchases <= 0:
        recs.append(
            Recommendation(
                title="Use non-monetary rewards to re-activate (cosmetics, boosts, convenience)",
                rationale="For non-spenders, value-forward rewards can boost engagement without pushing paywalls.",
                expected_impact="Increase engagement score proxy and progression momentum.",
                effort="Low",
                risk="Low (ensure fairness; avoid manipulative dark patterns).",
                metrics_to_track=["engagement_score_proxy", "returning_users_7d", "reward_claim_rate"],
                references=[refs[1]],
            )
        )

    # If we somehow produced nothing, emit a safe default.
    if not recs:
        recs.append(
            Recommendation(
                title="Run a gentle check-in + goal-setting prompt",
                rationale="When telemetry is sparse, a player-facing preference prompt can collect intent and improve personalization.",
                expected_impact="Improve personalization signals and reduce silent churn.",
                effort="Low",
                risk="Low (be transparent and allow skipping).",
                metrics_to_track=["prompt_completion_rate", "next_session_rate", "preference_signal_coverage"],
                references=refs,
            )
        )

    # In high risk, prioritize the top 3 to keep the plan focused.
    if risk_level == "High":
        return recs[:3]
    return recs[:4]


def _disclaimers() -> List[str]:
    return [
        "This report provides engagement optimization ideas based on limited gameplay signals and a predictive model; it is not ground truth.",
        "Avoid manipulative or coercive patterns (dark patterns). Provide clear opt-outs for notifications, personalization, and rewards.",
        "Validate recommendations via A/B tests and monitor for harm (e.g., increased spending pressure, addiction risk, frustration).",
        "Be mindful of fairness: do not disadvantage specific demographic groups or locations; audit outcomes by segment when possible.",
    ]


def generate_engagement_optimization_report(
    *,
    player_features: Mapping[str, Any],
    churn_probability: float,
    player_identifier: Optional[str] = None,
) -> Tuple[EngagementOptimizationReport, AgentState]:
    """Run the agent workflow and return (report, state)."""

    state = AgentState(
        step=AgentStep.INGEST,
        churn_probability=float(churn_probability),
        risk_level=_risk_bucket(float(churn_probability)),
        player_identifier=player_identifier,
        player_features=dict(player_features),
    )
    state.log("Ingested player features and churn probability.", {"has_player_id": bool(player_identifier)})

    state.step = AgentStep.ANALYZE
    profile = _build_engagement_profile(state)
    state.engagement_profile = {k: v for k, v in profile.items() if k != "_data_quality_notes"}
    state.data_quality_notes.extend(profile.get("_data_quality_notes", []))
    state.log("Derived engagement profile.", {"missing_notes": len(profile.get("_data_quality_notes", []))})

    state.step = AgentStep.RECOMMEND
    recs = _recommendations(profile, float(churn_probability), state.risk_level or "Medium")
    state.log("Generated recommendations.", {"count": len(recs)})

    state.step = AgentStep.FINALIZE
    base_refs = _base_references()
    report = EngagementOptimizationReport(
        generated_at_utc=EngagementOptimizationReport.now_utc_iso(),
        player_identifier=player_identifier,
        player_behavior_summary=_behavior_summary(profile),
        churn_risk_interpretation={
            "churn_probability": float(churn_probability),
            "risk_bucket": state.risk_level,
            "interpretation": (
                "Higher probability indicates higher predicted churn risk. Treat as a prioritization signal and validate with real retention outcomes."
            ),
        },
        engagement_and_retention_recommendations=recs,
        supporting_references=base_refs,
        ethical_and_ux_disclaimers=_disclaimers(),
        data_quality_notes=list(state.data_quality_notes),
    )
    state.log("Finalized report.")

    return report, state

