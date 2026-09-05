from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from .hybrid import repair_knapsack_sample
from .problems import KnapsackQUBO, knapsack_to_qubo, random_knapsack, random_maxcut
from .samplers import anneal_then_polish, exact_sample, simulated_annealing


@dataclass(frozen=True)
class MethodMetrics:
    mean_relative_gap: float
    max_relative_gap: float
    exact_hit_rate: float
    feasibility_rate: float


def _relative_gap(best_value: float, candidate_value: float) -> float:
    return float(max(0.0, best_value - candidate_value) / max(1.0, abs(best_value)))


def _summarize(gaps: list[float], hits: list[bool], feasible: list[bool]) -> MethodMetrics:
    return MethodMetrics(
        mean_relative_gap=float(np.mean(gaps)),
        max_relative_gap=float(np.max(gaps)),
        exact_hit_rate=float(np.mean(hits)),
        feasibility_rate=float(np.mean(feasible)),
    )


def benchmark_maxcut(
    *,
    instances: int = 6,
    n_nodes: int = 14,
    seed: int = 1000,
    num_reads: int = 24,
    sweeps: int = 220,
) -> dict[str, object]:
    sa_gaps: list[float] = []
    hybrid_gaps: list[float] = []
    sa_hits: list[bool] = []
    hybrid_hits: list[bool] = []
    for k in range(instances):
        problem = random_maxcut(n_nodes, 0.35, seed + k)
        qubo = problem.to_qubo()
        exact = exact_sample(qubo)
        optimum = problem.cut_value(exact.sample)
        sa = simulated_annealing(
            qubo,
            num_reads=num_reads,
            sweeps=sweeps,
            seed=seed + 10_000 + k,
        )[0]
        hybrid = anneal_then_polish(
            qubo,
            num_reads=num_reads,
            sweeps=sweeps,
            seed=seed + 20_000 + k,
        )
        sa_value = problem.cut_value(sa.sample)
        hybrid_value = problem.cut_value(hybrid.sample)
        sa_gap = _relative_gap(optimum, sa_value)
        hybrid_gap = _relative_gap(optimum, hybrid_value)
        sa_gaps.append(sa_gap)
        hybrid_gaps.append(hybrid_gap)
        sa_hits.append(sa_gap <= 1e-12)
        hybrid_hits.append(hybrid_gap <= 1e-12)
    return {
        "problem": "weighted-max-cut",
        "instances": instances,
        "n_nodes": n_nodes,
        "simulated_annealing": asdict(_summarize(sa_gaps, sa_hits, [True] * instances)),
        "sa_plus_polish": asdict(
            _summarize(hybrid_gaps, hybrid_hits, [True] * instances)
        ),
    }


def _knapsack_candidate(encoded: KnapsackQUBO, sample: np.ndarray) -> tuple[int, bool]:
    decoded = encoded.decode(sample)
    return int(decoded["profit"]), bool(decoded["feasible"] and decoded["equality_satisfied"])


def benchmark_knapsack(
    *,
    instances: int = 6,
    n_items: int = 9,
    seed: int = 2000,
    num_reads: int = 24,
    sweeps: int = 220,
) -> dict[str, object]:
    sa_gaps: list[float] = []
    hybrid_gaps: list[float] = []
    sa_hits: list[bool] = []
    hybrid_hits: list[bool] = []
    sa_feasible: list[bool] = []
    hybrid_feasible: list[bool] = []
    for k in range(instances):
        encoded = knapsack_to_qubo(random_knapsack(n_items, seed + k))
        exact = exact_sample(encoded.qubo)
        optimum, optimum_feasible = _knapsack_candidate(encoded, exact.sample)
        if not optimum_feasible:
            raise AssertionError("safe-penalty exact QUBO optimum must be feasible")
        sa = simulated_annealing(
            encoded.qubo,
            num_reads=num_reads,
            sweeps=sweeps,
            seed=seed + 10_000 + k,
        )[0]
        hybrid = anneal_then_polish(
            encoded.qubo,
            num_reads=num_reads,
            sweeps=sweeps,
            seed=seed + 20_000 + k,
        )
        repaired = repair_knapsack_sample(encoded, hybrid.sample)
        sa_profit, sa_ok = _knapsack_candidate(encoded, sa.sample)
        hybrid_profit, hybrid_ok = _knapsack_candidate(encoded, repaired)
        sa_gap = _relative_gap(optimum, sa_profit) if sa_ok else 1.0
        hybrid_gap = _relative_gap(optimum, hybrid_profit)
        sa_gaps.append(sa_gap)
        hybrid_gaps.append(hybrid_gap)
        sa_hits.append(sa_ok and sa_gap <= 1e-12)
        hybrid_hits.append(hybrid_ok and hybrid_gap <= 1e-12)
        sa_feasible.append(sa_ok)
        hybrid_feasible.append(hybrid_ok)
    return {
        "problem": "0-1-knapsack-qubo",
        "instances": instances,
        "n_items": n_items,
        "simulated_annealing": asdict(_summarize(sa_gaps, sa_hits, sa_feasible)),
        "sa_polish_plus_repair": asdict(
            _summarize(hybrid_gaps, hybrid_hits, hybrid_feasible)
        ),
    }
