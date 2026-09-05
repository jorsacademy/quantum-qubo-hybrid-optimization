from __future__ import annotations

import numpy as np

from .problems import KnapsackQUBO, encode_bounded_integer


def repair_knapsack_sample(encoded: KnapsackQUBO, sample: np.ndarray) -> np.ndarray:
    """Project an arbitrary QUBO sample to a feasible knapsack assignment.

    The repair removes low profit/weight items until feasible, then greedily adds high-ratio
    items that fit. Slack bits are recomputed exactly, so the QUBO capacity equality holds.
    """
    x = encoded.qubo.validate_sample(sample)
    items = x[: encoded.problem.n_items].copy()
    ratios = encoded.problem.profits / encoded.problem.weights

    while encoded.problem.total_weight(items) > encoded.problem.capacity:
        selected = np.flatnonzero(items)
        remove = int(selected[np.argmin(ratios[selected])])
        items[remove] = 0

    for index in np.argsort(ratios)[::-1]:
        if items[index] == 0:
            candidate_weight = encoded.problem.total_weight(items) + int(
                encoded.problem.weights[index]
            )
            if candidate_weight <= encoded.problem.capacity:
                items[index] = 1

    slack = encoded.problem.capacity - encoded.problem.total_weight(items)
    slack_bits = encode_bounded_integer(slack, encoded.slack_weights)
    return np.concatenate([items, slack_bits]).astype(np.int8)
