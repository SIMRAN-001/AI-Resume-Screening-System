"""Evaluate the ranking against human relevance labels (precision@k, recall@k, NDCG@k)."""
from __future__ import annotations

import math


def precision_at_k(ranked: list[str], relevance: dict[str, int], k: int, rel_threshold: int = 2) -> float:
    top = ranked[:k]
    return sum(1 for f in top if relevance.get(f, 0) >= rel_threshold) / max(k, 1)


def recall_at_k(ranked: list[str], relevance: dict[str, int], k: int, rel_threshold: int = 2) -> float:
    total_rel = sum(1 for v in relevance.values() if v >= rel_threshold)
    if total_rel == 0:
        return 0.0
    return sum(1 for f in ranked[:k] if relevance.get(f, 0) >= rel_threshold) / total_rel


def ndcg_at_k(ranked: list[str], relevance: dict[str, int], k: int) -> float:
    def dcg(items):
        return sum((2 ** relevance.get(f, 0) - 1) / math.log2(i + 2) for i, f in enumerate(items))

    ideal = sorted(relevance, key=relevance.get, reverse=True)
    best = dcg(ideal[:k])
    return dcg(ranked[:k]) / best if best else 0.0


def evaluate(ranked: list[str], relevance: dict[str, int], k: int = 5) -> dict[str, float]:
    return {
        f"precision@{k}": round(precision_at_k(ranked, relevance, k), 3),
        f"recall@{k}": round(recall_at_k(ranked, relevance, k), 3),
        f"ndcg@{k}": round(ndcg_at_k(ranked, relevance, k), 3),
    }
