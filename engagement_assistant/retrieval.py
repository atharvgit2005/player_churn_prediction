from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


@dataclass(frozen=True)
class StrategyEntry:
    title: str
    strategy: str
    when_to_use: str
    source: str
    url: Optional[str] = None
    tags: List[str] = field(default_factory=list)


@dataclass(frozen=True)
class RetrievedStrategy:
    title: str
    strategy: str
    when_to_use: str
    source: str
    url: Optional[str] = None
    score: float = 0.0
    matched_signals: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)


def _kb_path() -> Path:
    return Path(__file__).with_name("knowledge_base.jsonl")


@lru_cache(maxsize=1)
def load_strategy_knowledge_base(path: Optional[str] = None) -> List[StrategyEntry]:
    kb_file = Path(path) if path else _kb_path()
    if not kb_file.exists():
        return []

    strategies: List[StrategyEntry] = []
    with kb_file.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            record = json.loads(line)
            strategies.append(
                StrategyEntry(
                    title=str(record.get("title", "")).strip(),
                    strategy=str(record.get("strategy", "")).strip(),
                    when_to_use=str(record.get("when_to_use", "")).strip(),
                    source=str(record.get("source", "")).strip(),
                    url=(str(record["url"]).strip() if record.get("url") else None),
                    tags=[str(tag).strip().lower() for tag in record.get("tags", []) if str(tag).strip()],
                )
            )
    return strategies


def _join_text(parts: Iterable[str]) -> str:
    return " ".join(part for part in parts if part)


def build_retrieval_query(analysis: Mapping[str, Any]) -> str:
    risk_summary = analysis.get("risk_summary", {})
    if not isinstance(risk_summary, Mapping):
        risk_summary = {}

    parts: List[str] = []
    parts.append(str(analysis.get("risk_level", "")))
    parts.append(str(analysis.get("risk_label", "")))
    parts.append(str(analysis.get("risk_text", "")))
    parts.append(str(risk_summary.get("risk_level", "")))
    parts.append(str(risk_summary.get("risk_text", "")))
    parts.extend(str(item) for item in analysis.get("signal_phrases", []))
    parts.extend(str(item) for item in analysis.get("signal_tags", []))
    context = analysis.get("context", {})
    if isinstance(context, Mapping):
        parts.extend(str(v) for v in context.values() if v not in (None, ""))
    return _join_text(parts)


def _strategy_corpus(strategy: StrategyEntry) -> str:
    return _join_text(
        [
            strategy.title,
            strategy.strategy,
            strategy.when_to_use,
            strategy.source,
            " ".join(strategy.tags),
        ]
    )


def _tag_overlap_bonus(strategy: StrategyEntry, analysis_tags: Sequence[str]) -> float:
    if not strategy.tags or not analysis_tags:
        return 0.0

    strategy_tags = {tag.lower() for tag in strategy.tags}
    analysis_tag_set = {tag.lower() for tag in analysis_tags}
    overlap = strategy_tags.intersection(analysis_tag_set)
    return 0.12 * len(overlap)


def _risk_alignment_bonus(strategy: StrategyEntry, risk_level: str) -> float:
    risk = str(risk_level).strip().lower()
    tags = {tag.lower() for tag in strategy.tags}
    if risk == "high" and ({"at_risk", "high_risk", "return_hook"} & tags):
        return 0.16
    if risk == "medium" and ({"medium_risk", "progression", "habit"} & tags):
        return 0.08
    if risk == "low" and ({"habit", "retention"} & tags):
        return 0.06
    return 0.0


def rank_strategies(
    analysis: Mapping[str, Any],
    strategies: Optional[Sequence[StrategyEntry]] = None,
    top_k: int = 4,
) -> List[RetrievedStrategy]:
    kb = list(strategies) if strategies is not None else load_strategy_knowledge_base()
    if not kb:
        return []

    query = build_retrieval_query(analysis)
    corpus = [_strategy_corpus(strategy) for strategy in kb]
    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(corpus + [query or "engagement churn retention"])

    docs = matrix[:-1]
    query_vec = matrix[-1]
    similarities = cosine_similarity(docs, query_vec).ravel()

    signal_tags = [str(tag).strip().lower() for tag in analysis.get("signal_tags", []) if str(tag).strip()]
    signal_phrases = [str(phrase).strip() for phrase in analysis.get("signal_phrases", []) if str(phrase).strip()]
    risk_summary = analysis.get("risk_summary", {})
    if not isinstance(risk_summary, Mapping):
        risk_summary = {}
    risk_level = str(analysis.get("risk_level") or risk_summary.get("risk_level", "")).strip()

    ranked: List[RetrievedStrategy] = []
    for index, strategy in enumerate(kb):
        score = float(similarities[index])
        score += _tag_overlap_bonus(strategy, signal_tags)
        score += _risk_alignment_bonus(strategy, risk_level)
        matched_signal_set = {tag for tag in signal_tags if tag in {entry.lower() for entry in strategy.tags}}
        matched_signals = [phrase for tag, phrase in zip(signal_tags, signal_phrases) if tag in matched_signal_set]
        ranked.append(
            RetrievedStrategy(
                title=strategy.title,
                strategy=strategy.strategy,
                when_to_use=strategy.when_to_use,
                source=strategy.source,
                url=strategy.url,
                score=score,
                matched_signals=matched_signals,
                tags=list(strategy.tags),
            )
        )

    ranked.sort(key=lambda item: (item.score, item.title.lower()), reverse=True)
    return ranked[:top_k]
