from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import QUBO


@dataclass(frozen=True)
class SampleResult:
    sample: np.ndarray
    energy: float
    method: str


def exact_sample(qubo: QUBO, *, max_variables: int = 24) -> SampleResult:
    if qubo.n_variables > max_variables:
        raise ValueError(f"exact enumeration limited to {max_variables} variables")
    best_sample = np.zeros(qubo.n_variables, dtype=np.int8)
    best_energy = qubo.energy(best_sample)
    for integer in range(1, 1 << qubo.n_variables):
        sample = np.fromiter(
            ((integer >> i) & 1 for i in range(qubo.n_variables)),
            dtype=np.int8,
            count=qubo.n_variables,
        )
        energy = qubo.energy(sample)
        if energy < best_energy - 1e-12:
            best_sample = sample
            best_energy = energy
    return SampleResult(best_sample, float(best_energy), "exact-enumeration")


def _default_beta_range(qubo: QUBO) -> tuple[float, float]:
    interaction_scale = np.max(np.abs(qubo.linear) + np.sum(np.abs(qubo.quadratic), axis=1))
    scale = max(1.0, float(interaction_scale))
    return 0.1 / scale, 12.0 / scale


def simulated_annealing(
    qubo: QUBO,
    *,
    num_reads: int = 32,
    sweeps: int = 300,
    seed: int = 0,
    beta_range: tuple[float, float] | None = None,
) -> list[SampleResult]:
    if num_reads <= 0 or sweeps <= 0:
        raise ValueError("num_reads and sweeps must be positive")
    rng = np.random.default_rng(seed)
    beta_start, beta_end = beta_range or _default_beta_range(qubo)
    betas = np.linspace(beta_start, beta_end, sweeps)
    results: list[SampleResult] = []
    for _ in range(num_reads):
        sample = rng.integers(0, 2, size=qubo.n_variables, dtype=np.int8)
        energy = qubo.energy(sample)
        for beta in betas:
            for index in rng.permutation(qubo.n_variables):
                delta = qubo.flip_delta(sample, int(index))
                if delta <= 0.0 or rng.random() < np.exp(-beta * delta):
                    sample[index] = 1 - sample[index]
                    energy += delta
        energy = qubo.energy(sample)
        results.append(SampleResult(sample.copy(), float(energy), "simulated-annealing"))
    results.sort(key=lambda result: result.energy)
    return results


def one_flip_descent(qubo: QUBO, sample: np.ndarray) -> SampleResult:
    x = qubo.validate_sample(sample).copy()
    while True:
        deltas = np.array([qubo.flip_delta(x, i) for i in range(qubo.n_variables)])
        index = int(np.argmin(deltas))
        if deltas[index] >= -1e-12:
            break
        x[index] = 1 - x[index]
    return SampleResult(x, qubo.energy(x), "one-flip-descent")


def anneal_then_polish(
    qubo: QUBO,
    *,
    num_reads: int = 32,
    sweeps: int = 300,
    seed: int = 0,
    polish_candidates: int = 8,
) -> SampleResult:
    reads = simulated_annealing(
        qubo,
        num_reads=num_reads,
        sweeps=sweeps,
        seed=seed,
    )
    polished = [
        one_flip_descent(qubo, result.sample)
        for result in reads[: max(1, min(polish_candidates, len(reads)))]
    ]
    best = min(polished, key=lambda result: result.energy)
    return SampleResult(best.sample, best.energy, "sa-plus-one-flip-polish")
