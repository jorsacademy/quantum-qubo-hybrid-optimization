from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import QUBO


@dataclass(frozen=True)
class MaxCutProblem:
    n_nodes: int
    edges: tuple[tuple[int, int, float], ...]

    def cut_value(self, sample: np.ndarray) -> float:
        x = np.asarray(sample, dtype=np.int8)
        if x.shape != (self.n_nodes,):
            raise ValueError("sample has wrong shape")
        return float(sum(w for i, j, w in self.edges if int(x[i]) != int(x[j])))

    def to_qubo(self) -> QUBO:
        linear = np.zeros(self.n_nodes, dtype=float)
        quadratic = np.zeros((self.n_nodes, self.n_nodes), dtype=float)
        for i, j, weight in self.edges:
            linear[i] -= weight
            linear[j] -= weight
            quadratic[i, j] += 2.0 * weight
            quadratic[j, i] += 2.0 * weight
        return QUBO(
            linear=linear,
            quadratic=quadratic,
            labels=tuple(f"side_{i}" for i in range(self.n_nodes)),
        )


@dataclass(frozen=True)
class KnapsackProblem:
    profits: np.ndarray
    weights: np.ndarray
    capacity: int

    def __post_init__(self) -> None:
        profits = np.asarray(self.profits, dtype=int)
        weights = np.asarray(self.weights, dtype=int)
        if profits.ndim != 1 or weights.shape != profits.shape:
            raise ValueError("profits and weights must be one-dimensional with equal length")
        if np.any(profits <= 0) or np.any(weights <= 0) or self.capacity <= 0:
            raise ValueError("profits, weights, and capacity must be positive")
        object.__setattr__(self, "profits", profits)
        object.__setattr__(self, "weights", weights)

    @property
    def n_items(self) -> int:
        return int(self.profits.size)

    def value(self, items: np.ndarray) -> int:
        x = np.asarray(items, dtype=np.int8)
        if x.shape != (self.n_items,):
            raise ValueError("item sample has wrong shape")
        return int(self.profits @ x)

    def total_weight(self, items: np.ndarray) -> int:
        x = np.asarray(items, dtype=np.int8)
        return int(self.weights @ x)

    def feasible(self, items: np.ndarray) -> bool:
        return self.total_weight(items) <= self.capacity


@dataclass(frozen=True)
class KnapsackQUBO:
    problem: KnapsackProblem
    qubo: QUBO
    slack_weights: np.ndarray
    penalty: float

    def decode(self, sample: np.ndarray) -> dict[str, object]:
        x = self.qubo.validate_sample(sample)
        items = x[: self.problem.n_items]
        slack_bits = x[self.problem.n_items :]
        weight = self.problem.total_weight(items)
        slack = int(self.slack_weights @ slack_bits)
        return {
            "items": items.copy(),
            "profit": self.problem.value(items),
            "weight": weight,
            "slack": slack,
            "feasible": weight <= self.problem.capacity,
            "equality_satisfied": weight + slack == self.problem.capacity,
        }


def bounded_binary_weights(max_value: int) -> np.ndarray:
    """Weights that can represent every integer from zero through max_value."""
    if max_value < 0:
        raise ValueError("max_value must be nonnegative")
    if max_value == 0:
        return np.zeros(0, dtype=int)
    result: list[int] = []
    represented = 0
    power = 1
    while represented < max_value:
        weight = min(power, max_value - represented)
        result.append(weight)
        represented += weight
        power *= 2
    return np.asarray(result, dtype=int)


def encode_bounded_integer(value: int, weights: np.ndarray) -> np.ndarray:
    weights = np.asarray(weights, dtype=int)
    if value < 0 or value > int(weights.sum()):
        raise ValueError("value is outside representable range")
    bits = np.zeros(weights.size, dtype=np.int8)
    remaining = int(value)
    for index in np.argsort(weights)[::-1]:
        weight = int(weights[index])
        if weight <= remaining:
            bits[index] = 1
            remaining -= weight
    if remaining != 0:
        raise RuntimeError("bounded binary encoding failed")
    return bits


def knapsack_to_qubo(
    problem: KnapsackProblem,
    *,
    penalty: float | None = None,
) -> KnapsackQUBO:
    """Encode integer knapsack capacity with binary slack and a squared penalty.

    With positive integer profits/weights, P > sum(profits) is a sufficient penalty:
    any equality-violating binary assignment has penalty at least P, while its objective
    contribution is bounded below by -sum(profits). A feasible all-zero item assignment
    exists, so a global QUBO minimizer must satisfy the capacity equality.
    """
    p = float(problem.profits.sum() + 1 if penalty is None else penalty)
    if p <= float(problem.profits.sum()):
        raise ValueError("penalty must exceed sum(profits) for the documented guarantee")

    slack_weights = bounded_binary_weights(problem.capacity)
    coefficients = np.concatenate([problem.weights, slack_weights]).astype(float)
    n = coefficients.size
    linear = p * (coefficients**2 - 2.0 * problem.capacity * coefficients)
    linear[: problem.n_items] -= problem.profits
    quadratic = np.zeros((n, n), dtype=float)
    for i in range(n):
        for j in range(i + 1, n):
            coefficient = 2.0 * p * coefficients[i] * coefficients[j]
            quadratic[i, j] = coefficient
            quadratic[j, i] = coefficient
    labels = tuple(f"item_{i}" for i in range(problem.n_items)) + tuple(
        f"slack_{i}" for i in range(slack_weights.size)
    )
    qubo = QUBO(
        linear=linear,
        quadratic=quadratic,
        offset=p * float(problem.capacity**2),
        labels=labels,
    )
    return KnapsackQUBO(problem=problem, qubo=qubo, slack_weights=slack_weights, penalty=p)


def random_maxcut(n_nodes: int, edge_probability: float, seed: int) -> MaxCutProblem:
    if n_nodes < 2 or not 0.0 < edge_probability <= 1.0:
        raise ValueError("invalid graph parameters")
    rng = np.random.default_rng(seed)
    edges: list[tuple[int, int, float]] = []
    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if rng.random() <= edge_probability:
                edges.append((i, j, float(rng.integers(1, 10))))
    if not edges:
        edges.append((0, 1, 1.0))
    return MaxCutProblem(n_nodes=n_nodes, edges=tuple(edges))


def random_knapsack(n_items: int, seed: int) -> KnapsackProblem:
    if n_items < 2:
        raise ValueError("n_items must be at least two")
    rng = np.random.default_rng(seed)
    weights = rng.integers(1, 10, size=n_items)
    profits = rng.integers(2, 25, size=n_items)
    capacity = max(1, int(np.floor(0.45 * weights.sum())))
    return KnapsackProblem(profits=profits, weights=weights, capacity=capacity)
