from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .model import QUBO


@dataclass(frozen=True)
class DWaveResult:
    sample: np.ndarray
    energy: float
    solver: str
    info: dict[str, object]


def _imports() -> tuple[object, object, object, object]:
    try:
        import dimod
        from dwave.system import DWaveSampler, EmbeddingComposite, LeapHybridSampler
    except ImportError as exc:
        raise RuntimeError(
            'D-Wave integration is optional. Install with: pip install -e ".[dwave]"'
        ) from exc
    return dimod, DWaveSampler, EmbeddingComposite, LeapHybridSampler


def to_dimod_bqm(qubo: QUBO) -> object:
    dimod, _, _, _ = _imports()
    linear = {label: float(qubo.linear[i]) for i, label in enumerate(qubo.labels)}
    quadratic = {
        (qubo.labels[i], qubo.labels[j]): float(qubo.quadratic[i, j])
        for i in range(qubo.n_variables)
        for j in range(i + 1, qubo.n_variables)
        if abs(qubo.quadratic[i, j]) > 0.0
    }
    return dimod.BinaryQuadraticModel(linear, quadratic, qubo.offset, dimod.BINARY)


def _sample_array(qubo: QUBO, sample_mapping: object) -> np.ndarray:
    return np.asarray([int(sample_mapping[label]) for label in qubo.labels], dtype=np.int8)


def sample_leap_hybrid(
    qubo: QUBO,
    *,
    time_limit: int | None = None,
    label: str = "Jors Academy - QUBO hybrid benchmark",
) -> DWaveResult:
    _, _, _, LeapHybridSampler = _imports()
    bqm = to_dimod_bqm(qubo)
    with LeapHybridSampler() as sampler:
        kwargs: dict[str, object] = {"label": label}
        if time_limit is not None:
            kwargs["time_limit"] = int(time_limit)
        sampleset = sampler.sample(bqm, **kwargs)
    sample = _sample_array(qubo, sampleset.first.sample)
    return DWaveResult(
        sample=sample,
        energy=qubo.energy(sample),
        solver="dwave-leap-hybrid-bqm",
        info=dict(sampleset.info),
    )


def sample_qpu(
    qubo: QUBO,
    *,
    num_reads: int = 100,
) -> DWaveResult:
    _, DWaveSampler, EmbeddingComposite, _ = _imports()
    bqm = to_dimod_bqm(qubo)
    sampler = EmbeddingComposite(DWaveSampler())
    sampleset = sampler.sample(bqm, num_reads=int(num_reads))
    sample = _sample_array(qubo, sampleset.first.sample)
    return DWaveResult(
        sample=sample,
        energy=qubo.energy(sample),
        solver="dwave-qpu-embedded",
        info=dict(sampleset.info),
    )
