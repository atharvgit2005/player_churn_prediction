from __future__ import annotations

from dataclasses import asdict, replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .retrieval import RetrievedStrategy, StrategyEntry, load_strategy_knowledge_base, rank_strategies
from .schema import EngagementOptimizationReport, Reference, Recommendation
from .state import AgentState, AgentStep


def _safe_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        if isinstance(value, str) and not value.strip():
            return None
        number = float(value)
        if np.isnan(number) or np.isinf(number):
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


def _risk_text(risk_level: str, probability: float) -> str:
    if risk_level == "High":
        return f"Churn probability is {probability:.1%}, which places this player in a high-risk bucket."
    if risk_level == "Medium":
        return f"Churn probability is {probability:.1%}, which suggests moderate retention risk."
    return f"Churn probability is {probability:.1%}, which suggests a lower immediate churn risk."


def _unique_preserve_order(values: Sequence[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            ordered.append(value)
    return ordered


def _add_signal(signal_tags: List[str], signal_phrases: List[str], tag: str, phrase: str) -> None:
    if tag not in signal_tags:
        signal_tags.append(tag)
    if phrase not in signal_phrases:
        signal_phrases.append(phrase)


def _build_player_analysis(state: AgentState) -> Dict[str, Any]:
    feats = state.player_features
    notes: List[str] = []
    signal_tags: List[str] = []
    signal_phrases: List[str] = []

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
        notes.append("Sessions per week is missing; frequency-based recommendations are less certain.")
    elif sessions_f is None:
        notes.append("Sessions per week could not be parsed as a number.")
    elif sessions_f < 3:
        _add_signal(signal_tags, signal_phrases, "low_sessions", f"Weekly sessions are low ({sessions_f:.1f}).")
    elif sessions_f >= 6:
        _add_signal(signal_tags, signal_phrases, "habit", f"Weekly sessions are frequent ({sessions_f:.1f}).")

    if avg_minutes is None:
        notes.append("Average session duration is missing; session-length recommendations are less certain.")
    elif avg_minutes_f is None:
        notes.append("Average session duration could not be parsed as a number.")
    elif avg_minutes_f < 10:
        _add_signal(signal_tags, signal_phrases, "short_sessions", f"Average session duration is short ({avg_minutes_f:.1f} minutes).")
    elif avg_minutes_f >= 30:
        _add_signal(signal_tags, signal_phrases, "deep_sessions", f"Average session duration is long ({avg_minutes_f:.1f} minutes).")

    if play_hours is None:
        notes.append("Play time hours is missing; overall engagement depth is less certain.")
    elif play_hours_f is None:
        notes.append("Play time hours could not be parsed as a number.")
    elif play_hours_f < 5:
        _add_signal(signal_tags, signal_phrases, "light_playtime", f"Total play time is low ({play_hours_f:.1f} hours).")

    if purchases is None:
        notes.append("In-game purchases is missing; monetization-friendly strategies are conservative by default.")
    elif purchases_f is None:
        notes.append("In-game purchases could not be parsed as a number.")
    elif purchases_f <= 0:
        _add_signal(signal_tags, signal_phrases, "low_spend", "The player has no recent in-game purchases.")
    else:
        _add_signal(signal_tags, signal_phrases, "purchasing", "The player has made in-game purchases.")

    if level is not None and level_f is not None:
        if level_f < 10:
            _add_signal(signal_tags, signal_phrases, "early_progression", f"Player level is still early ({level_f:.1f}).")
        elif level_f >= 30:
            _add_signal(signal_tags, signal_phrases, "advanced_progression", f"Player level is high ({level_f:.1f}).")
    elif level is None:
        notes.append("Player level is missing; progression-based recommendations are less certain.")

    if achievements is not None and achievements_f is not None:
        if achievements_f < 5:
            _add_signal(signal_tags, signal_phrases, "slow_progress", f"Achievement count is low ({achievements_f:.1f}).")
        elif achievements_f >= 20:
            _add_signal(signal_tags, signal_phrases, "achievement_rich", f"Achievement count is high ({achievements_f:.1f}).")
    elif achievements is None:
        notes.append("Achievements unlocked is missing; progression confidence is lower.")

    difficulty_text = str(difficulty or "").strip().lower()
    if difficulty is None:
        notes.append("Game difficulty is missing; difficulty-sensitive recommendations are less certain.")
    elif difficulty_text in {"hard", "high", "expert", "challenging"}:
        _add_signal(signal_tags, signal_phrases, "high_difficulty", f"Game difficulty is set to {difficulty}.")
    elif difficulty_text in {"easy", "casual"}:
        _add_signal(signal_tags, signal_phrases, "low_difficulty", f"Game difficulty is set to {difficulty}.")

    genre_text = str(genre or "").strip().lower()
    if genre_text:
        if any(token in genre_text for token in ["multiplayer", "coop", "co-op", "social", "mmo", "guild", "team"]):
            _add_signal(signal_tags, signal_phrases, "social_play", f"Game genre suggests social play: {genre}.")
        if any(token in genre_text for token in ["strategy", "rpg", "simulation"]):
            _add_signal(signal_tags, signal_phrases, "progression_focused", f"Game genre suggests long-form progression: {genre}.")

    engagement_text = str(engagement_level or "").strip().lower()
    if engagement_level is None:
        notes.append("Observed engagement level is missing; the assistant relies more on other gameplay signals.")
    elif engagement_text in {"low", "poor", "weak"}:
        _add_signal(signal_tags, signal_phrases, "low_engagement", f"Observed engagement level is {engagement_level}.")
    elif engagement_text in {"high", "strong", "excellent"}:
        _add_signal(signal_tags, signal_phrases, "strong_engagement", f"Observed engagement level is {engagement_level}.")

    present_key_count = sum(
        1
        for value in [sessions, avg_minutes, play_hours, purchases, level, achievements, genre, difficulty, engagement_level]
        if value is not None
    )
    coverage_score = present_key_count / 9.0
    if coverage_score < 0.45:
        notes.append("Data coverage is sparse; recommendations will stay conservative and preference-driven.")
    elif coverage_score < 0.7:
        notes.append("Data coverage is partial; recommendations should be treated as directional guidance.")

    if len(notes) >= 4:
        notes.append("Several signals are missing or noisy, so uncertainty notes are included with the recommendations.")

    if not signal_tags:
        _add_signal(signal_tags, signal_phrases, "uncertain", "The available data is too sparse for a strong behavioral signal.")

    engagement_score = None
    if sessions_f is not None and avg_minutes_f is not None:
        engagement_score = sessions_f * avg_minutes_f

    behavior_summary = {
        "engagement_overview": {
            "sessions_per_week": sessions_f,
            "avg_session_duration_minutes": avg_minutes_f,
            "play_time_hours": play_hours_f,
            "engagement_score_proxy": engagement_score,
        },
        "progression_overview": {
            "player_level": level_f,
            "achievements_unlocked": achievements_f,
        },
        "monetization_overview": {
            "in_game_purchases": purchases_f,
        },
        "context": {
            "game_genre": genre,
            "game_difficulty": difficulty,
            "engagement_level_observed": engagement_level,
        },
        "signal_tags": signal_tags,
        "signal_phrases": signal_phrases,
    }

    return {
        "player_behavior_summary": behavior_summary,
        "signal_tags": signal_tags,
        "signal_phrases": signal_phrases,
        "data_quality_notes": _unique_preserve_order(notes),
        "coverage_score": round(coverage_score, 3),
        "confidence_level": "low" if coverage_score < 0.45 else "medium" if coverage_score < 0.7 else "high",
        "risk_summary": {
            "risk_probability": float(state.churn_probability or 0.0),
            "risk_level": state.risk_level or _risk_bucket(float(state.churn_probability or 0.0)),
            "risk_text": _risk_text(state.risk_level or _risk_bucket(float(state.churn_probability or 0.0)), float(state.churn_probability or 0.0)),
        },
        "context": {
            "game_genre": genre,
            "game_difficulty": difficulty,
            "engagement_level_observed": engagement_level,
        },
        "query_text": "",
    }


def _build_retrieval_query(analysis: Mapping[str, Any]) -> str:
    query_parts: List[str] = [
        str(analysis.get("risk_summary", {}).get("risk_level", "")),
        str(analysis.get("risk_summary", {}).get("risk_text", "")),
    ]
    query_parts.extend(str(item) for item in analysis.get("signal_phrases", []))
    query_parts.extend(str(item) for item in analysis.get("signal_tags", []))
    context = analysis.get("context", {})
    if isinstance(context, Mapping):
        query_parts.extend(str(value) for value in context.values() if value not in (None, ""))
    return " ".join(part for part in query_parts if part)


def _reference_from_strategy(strategy: RetrievedStrategy) -> Reference:
    note = strategy.when_to_use or strategy.strategy
    return Reference(title=strategy.title, source=strategy.source, note=note, url=strategy.url)


def _strategy_for_title(title: str, strategies: Sequence[RetrievedStrategy]) -> Optional[RetrievedStrategy]:
    needle = title.strip().lower()
    for strategy in strategies:
        if strategy.title.strip().lower() == needle:
            return strategy
    return None


def _metrics_for_strategy(strategy: RetrievedStrategy, analysis: Mapping[str, Any]) -> List[str]:
    tags = {tag.lower() for tag in strategy.tags}
    metrics: List[str] = []

    metric_map = [
        ({"low_sessions", "return_hook", "habit"}, ["time_to_next_session", "d7_retention", "return_rate"]),
        ({"short_sessions", "quick_win"}, ["avg_session_duration_minutes", "mission_completion_rate", "session_dropoff_rate"]),
        ({"high_difficulty", "frustration"}, ["fail_streak_length", "level_retry_rate", "quit_during_fail_streak"]),
        ({"slow_progress", "progression", "milestone"}, ["time_to_milestone", "unlock_rate", "progression_completion_rate"]),
        ({"social", "community", "cooperative"}, ["party_join_rate", "co_op_session_rate", "d7_retention"]),
        ({"non_monetary", "low_spend"}, ["reward_claim_rate", "returning_users_7d", "engagement_score_proxy"]),
        ({"sparse_data", "noisy_data", "preference"}, ["preference_signal_coverage", "prompt_completion_rate", "return_rate"]),
        ({"event", "limited_time"}, ["event_participation_rate", "return_rate", "retention_lift"]),
        ({"confusion", "onboarding"}, ["tutorial_completion_rate", "support_click_rate", "early_exit_rate"]),
    ]

    for required_tags, metric_candidates in metric_map:
        if tags.intersection(required_tags):
            metrics.extend(metric_candidates)

    if not metrics:
        metrics.extend(["d7_retention", "return_rate", "player_satisfaction"])

    if analysis.get("coverage_score", 1.0) < 0.7:
        metrics.append("data_quality_score")

    return _unique_preserve_order(metrics)


def _effort_for_strategy(strategy: RetrievedStrategy) -> str:
    tags = {tag.lower() for tag in strategy.tags}
    if tags.intersection({"quick_win", "return_hook", "low_spend", "preference"}):
        return "Low"
    if tags.intersection({"high_difficulty", "milestone", "progression", "onboarding", "social", "event"}):
        return "Medium"
    return "Medium"


def _risk_for_strategy(strategy: RetrievedStrategy, analysis: Mapping[str, Any]) -> str:
    risk_level = str(analysis.get("risk_summary", {}).get("risk_level", "Medium")).lower()
    tags = {tag.lower() for tag in strategy.tags}
    if risk_level == "high" and tags.intersection({"return_hook", "quick_win", "preference", "low_spend", "high_difficulty"}):
        return "Low risk if opt-outs and frequency caps are enforced."
    if tags.intersection({"social", "event"}):
        return "Medium risk; avoid over-messaging and measure opt-out rates."
    if tags.intersection({"high_difficulty", "frustration"}):
        return "Medium risk; keep the assist optional so skilled players keep autonomy."
    return "Low risk with standard measurement and clear opt-out."


def _action_steps(strategy: RetrievedStrategy, analysis: Mapping[str, Any]) -> List[str]:
    tags = {tag.lower() for tag in strategy.tags}
    steps: List[str] = []

    if tags.intersection({"low_sessions", "return_hook", "quick_win"}):
        steps.extend(
            [
                "Ship a single re-entry objective that can be finished in one short play session.",
                "Cap reminders to avoid spam and make opt-out visible.",
                "Track time_to_next_session, completion rate, and 7-day retention for the target segment.",
            ]
        )
    elif tags.intersection({"high_difficulty", "frustration"}):
        steps.extend(
            [
                "Add an optional assist or hint at the friction point without changing the core challenge.",
                "Leave the main difficulty untouched for skilled players and keep the support optional.",
                "Track fail streak length, retry rate, and quit during fail streak events.",
            ]
        )
    elif tags.intersection({"slow_progress", "milestone", "progression"}):
        steps.extend(
            [
                "Expose the next milestone clearly so the player can see the payoff before the next session ends.",
                "Tie the milestone to one measurable action and one reward.",
                "Track milestone completion, unlock rate, and progression completion time.",
            ]
        )
    elif tags.intersection({"social", "cooperative", "community"}):
        steps.extend(
            [
                "Invite the player into one shared objective or co-op action that can be completed quickly.",
                "Make the benefit visible to the player and their teammates.",
                "Track party join rate, co-op session rate, and 7-day retention.",
            ]
        )
    elif tags.intersection({"non_monetary", "low_spend"}):
        steps.extend(
            [
                "Offer a non-monetary reward path such as cosmetics, boosts, or convenience perks.",
                "Keep the reward transparent and tied to return behavior instead of spending pressure.",
                "Track reward claim rate, returning users, and engagement score proxy.",
            ]
        )
    elif tags.intersection({"sparse_data", "noisy_data", "preference"}):
        steps.extend(
            [
                "Ask one or two preference questions to improve future targeting.",
                "Explain why the question is being asked and allow the player to skip it.",
                "Track prompt completion rate, preference coverage, and the next-session rate.",
            ]
        )
    else:
        steps.extend(
            [
                "Deploy the strategy as a narrow A/B test for the identified player segment.",
                "Tie the rollout to a measurable retention or engagement metric.",
                "Monitor opt-outs, return rate, and any signs of unintended friction.",
            ]
        )

    if analysis.get("coverage_score", 1.0) < 0.7 and "preference" not in tags:
        steps.append("Add a short uncertainty-safe check-in because the available signals are partial.")

    return _unique_preserve_order(steps)


def _build_heuristic_recommendations(
    analysis: Mapping[str, Any],
    retrieved_strategies: Sequence[RetrievedStrategy],
) -> List[Recommendation]:
    recommendations: List[Recommendation] = []
    signal_phrases = list(analysis.get("signal_phrases", []))
    risk_summary = analysis.get("risk_summary", {})
    coverage_score = float(analysis.get("coverage_score", 1.0))
    uncertainty_notes = list(analysis.get("data_quality_notes", []))

    if not retrieved_strategies:
        fallback_reference = Reference(
            title="Ask for a preference check-in",
            source="Fallback safe strategy",
            note="Use when telemetry is sparse or noisy.",
            url=None,
        )
        recommendations.append(
            Recommendation(
                title=fallback_reference.title,
                rationale="The data is too sparse for a strong behavioral read, so a quick preference check-in is the safest next step.",
                expected_impact="Improve personalization signals and reduce silent churn risk.",
                effort="Low",
                risk="Low risk with a visible skip option and a transparent explanation.",
                metrics_to_track=["prompt_completion_rate", "preference_signal_coverage", "next_session_rate"],
                references=[fallback_reference],
                supporting_signals=["sparse_data"],
                action_steps=[
                    "Ask one or two preference questions and explain why they help personalize the experience.",
                    "Let the player skip the prompt without penalty.",
                    "Track completion rate and whether the extra signal changes the next-session rate.",
                ],
                confidence=0.35,
                uncertainty_notes=list(uncertainty_notes) or ["Sparse data means this is a conservative fallback recommendation."],
            )
        )
        return recommendations

    for strategy in retrieved_strategies[:4]:
        supporting_signals = strategy.matched_signals or list(signal_phrases[:1])
        if not supporting_signals:
            supporting_signals = [f"churn risk is {risk_summary.get('risk_level', 'unknown').lower()}"]

        recommendations.append(
            Recommendation(
                title=strategy.title,
                rationale=(
                    f"{strategy.when_to_use} The retrieval layer matched this strategy against the current player signals and churn risk."
                ),
                expected_impact=strategy.strategy,
                effort=_effort_for_strategy(strategy),
                risk=_risk_for_strategy(strategy, analysis),
                metrics_to_track=_metrics_for_strategy(strategy, analysis),
                references=[_reference_from_strategy(strategy)],
                supporting_signals=supporting_signals,
                action_steps=_action_steps(strategy, analysis),
                confidence=round(min(0.95, max(0.25, strategy.score + 0.15)), 3),
                uncertainty_notes=list(uncertainty_notes) if coverage_score < 0.7 else [],
            )
        )

    if coverage_score < 0.45 and recommendations:
        recommendations[0] = replace(
            recommendations[0],
            uncertainty_notes=_unique_preserve_order(
                list(recommendations[0].uncertainty_notes)
                + ["The available telemetry is sparse, so this recommendation should be treated as provisional."]
            ),
        )

    return recommendations


def _normalize_recommendation_payload(
    payload: Mapping[str, Any],
    retrieved_strategies: Sequence[RetrievedStrategy],
    analysis: Mapping[str, Any],
) -> Optional[Recommendation]:
    title = str(payload.get("title", "")).strip()
    rationale = str(payload.get("rationale", "")).strip()
    expected_impact = str(payload.get("expected_impact", "")).strip()
    effort = str(payload.get("effort", "Medium")).strip() or "Medium"
    risk = str(payload.get("risk", "")).strip() or "Low"

    if not title or not rationale or not expected_impact:
        return None

    reference_titles = [str(item).strip() for item in payload.get("reference_titles", []) if str(item).strip()]
    matched_strategies = [
        strategy
        for strategy in ( _strategy_for_title(name, retrieved_strategies) for name in reference_titles )
        if strategy is not None
    ]
    if not matched_strategies and retrieved_strategies:
        matched_strategies = [retrieved_strategies[0]]

    supporting_signals = [str(item).strip() for item in payload.get("supporting_signals", []) if str(item).strip()]
    if not supporting_signals:
        supporting_signals = list(analysis.get("signal_phrases", []))[:1]
    if not supporting_signals:
        supporting_signals = [f"churn risk is {analysis.get('risk_summary', {}).get('risk_level', 'unknown').lower()}"]

    metrics = [str(item).strip() for item in payload.get("metrics_to_track", []) if str(item).strip()]
    action_steps = [str(item).strip() for item in payload.get("action_steps", []) if str(item).strip()]
    uncertainty_notes = [str(item).strip() for item in payload.get("uncertainty_notes", []) if str(item).strip()]
    confidence = _safe_float(payload.get("confidence"))

    if not matched_strategies:
        return None

    references = [_reference_from_strategy(strategy) for strategy in matched_strategies]

    return Recommendation(
        title=title,
        rationale=rationale,
        expected_impact=expected_impact,
        effort=effort,
        risk=risk,
        metrics_to_track=_unique_preserve_order(metrics) or _metrics_for_strategy(matched_strategies[0], analysis),
        references=references,
        supporting_signals=_unique_preserve_order(supporting_signals),
        action_steps=_unique_preserve_order(action_steps) or _action_steps(matched_strategies[0], analysis),
        confidence=confidence if confidence is not None else round(min(0.95, max(0.25, matched_strategies[0].score + 0.1)), 3),
        uncertainty_notes=_unique_preserve_order(uncertainty_notes),
    )


def _supporting_references(retrieved_strategies: Sequence[RetrievedStrategy]) -> List[Reference]:
    references: List[Reference] = []
    seen = set()
    for strategy in retrieved_strategies:
        ref = _reference_from_strategy(strategy)
        key = (ref.title.lower(), ref.source.lower())
        if key not in seen:
            seen.add(key)
            references.append(ref)
    return references


def _finalize_recommendations(
    recommendations: Sequence[Recommendation],
    analysis: Mapping[str, Any],
    retrieved_strategies: Sequence[RetrievedStrategy],
) -> List[Recommendation]:
    final: List[Recommendation] = []
    allowed_titles = {strategy.title.lower() for strategy in retrieved_strategies}
    allowed_signals = {signal.lower() for signal in analysis.get("signal_tags", [])}
    allowed_signals.update(signal.lower() for signal in analysis.get("signal_phrases", []))
    risk_text = str(analysis.get("risk_summary", {}).get("risk_text", "")).strip()

    for recommendation in recommendations:
        if not recommendation.title or not recommendation.rationale or not recommendation.expected_impact:
            continue

        references = [ref for ref in recommendation.references if ref.title.lower() in allowed_titles]
        if not references and retrieved_strategies:
            references = [_reference_from_strategy(retrieved_strategies[0])]

        if not references:
            continue

        supporting_signals = [
            signal
            for signal in recommendation.supporting_signals
            if signal.lower() in allowed_signals or signal.lower().startswith("churn risk")
        ]
        if not supporting_signals and allowed_signals:
            supporting_signals = [next(iter(allowed_signals))]

        action_steps = recommendation.action_steps or [
            "Run a narrow A/B test.",
            "Measure the relevant engagement and retention metrics.",
            "Keep the rollout opt-in and reversible.",
        ]

        uncertainty_notes = list(recommendation.uncertainty_notes)
        if risk_text and risk_text not in uncertainty_notes and analysis.get("coverage_score", 1.0) < 0.7:
            uncertainty_notes.append(risk_text)

        final.append(
            Recommendation(
                title=recommendation.title,
                rationale=recommendation.rationale,
                expected_impact=recommendation.expected_impact,
                effort=recommendation.effort,
                risk=recommendation.risk,
                metrics_to_track=_unique_preserve_order(recommendation.metrics_to_track) or ["d7_retention", "return_rate"],
                references=references,
                supporting_signals=_unique_preserve_order(supporting_signals),
                action_steps=_unique_preserve_order(action_steps),
                confidence=recommendation.confidence,
                uncertainty_notes=_unique_preserve_order(uncertainty_notes),
            )
        )

    if final:
        return final

    fallback_strategy = retrieved_strategies[0] if retrieved_strategies else None
    if fallback_strategy is None:
        return []

    return [
        Recommendation(
            title=fallback_strategy.title,
            rationale=fallback_strategy.when_to_use or "Use the retrieved strategy as a conservative fallback.",
            expected_impact=fallback_strategy.strategy,
            effort=_effort_for_strategy(fallback_strategy),
            risk=_risk_for_strategy(fallback_strategy, analysis),
            metrics_to_track=_metrics_for_strategy(fallback_strategy, analysis),
            references=[_reference_from_strategy(fallback_strategy)],
            supporting_signals=list(analysis.get("signal_phrases", []))[:1] or ["churn risk elevated"],
            action_steps=_action_steps(fallback_strategy, analysis),
            confidence=round(min(0.95, max(0.25, fallback_strategy.score + 0.1)), 3),
            uncertainty_notes=[
                "The requested recommendation could not be fully validated, so this fallback stays close to the retrieved strategy."
            ],
        )
    ]


def _build_disclaimers() -> List[str]:
    return [
        "This report provides engagement optimization ideas based on limited gameplay signals and a predictive model; it is not ground truth.",
        "Avoid manipulative or coercive patterns. Provide clear opt-outs for notifications, personalization, and rewards.",
        "Validate recommendations via A/B tests and monitor for harm such as increased spending pressure, frustration, or fatigue.",
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
    state.log(
        "Ingested player features and churn probability.",
        {"has_player_id": bool(player_identifier), "churn_probability": round(float(churn_probability), 4)},
    )

    state.step = AgentStep.ANALYZE
    analysis = _build_player_analysis(state)
    analysis["query_text"] = _build_retrieval_query(analysis)
    state.analysis_summary = analysis
    state.engagement_profile = analysis["player_behavior_summary"]
    state.data_quality_notes.extend(analysis["data_quality_notes"])
    state.log(
        "Analyzed churn risk and engagement patterns.",
        {
            "risk_level": analysis["risk_summary"]["risk_level"],
            "coverage_score": analysis["coverage_score"],
            "signal_count": len(analysis["signal_tags"]),
        },
    )

    state.step = AgentStep.RETRIEVE
    kb = load_strategy_knowledge_base()
    retrieved_strategies = rank_strategies(analysis, kb, top_k=4)
    state.retrieved_strategies = [asdict(strategy) for strategy in retrieved_strategies]
    state.log(
        "Retrieved relevant engagement strategies.",
        {
            "knowledge_base_size": len(kb),
            "retrieved_count": len(retrieved_strategies),
            "top_titles": [strategy.title for strategy in retrieved_strategies],
        },
    )

    state.step = AgentStep.RECOMMEND
    heuristic_recommendations = _build_heuristic_recommendations(analysis, retrieved_strategies)
    state.recommendation_drafts = [asdict(recommendation) for recommendation in heuristic_recommendations]
    final_recommendations = _finalize_recommendations(heuristic_recommendations, analysis, retrieved_strategies)

    state.workflow_mode = "retrieval"
    state.log(
        "Generated recommendations.",
        {
            "mode": state.workflow_mode,
            "count": len(final_recommendations),
        },
    )

    state.step = AgentStep.FINALIZE
    supporting_references = _supporting_references(retrieved_strategies)
    report = EngagementOptimizationReport(
        generated_at_utc=EngagementOptimizationReport.now_utc_iso(),
        player_identifier=player_identifier,
        player_behavior_summary=analysis["player_behavior_summary"],
        churn_risk_interpretation=analysis["risk_summary"],
        engagement_and_retention_recommendations=final_recommendations,
        supporting_references=supporting_references,
        ethical_and_ux_disclaimers=_build_disclaimers(),
        data_quality_notes=analysis["data_quality_notes"],
        analysis_summary=analysis,
        retrieved_strategies=retrieved_strategies,
        workflow_mode=state.workflow_mode,
    )
    state.log(
        "Finalized report.",
        {
            "supporting_references": len(supporting_references),
            "recommendations": len(final_recommendations),
        },
    )

    return report, state
