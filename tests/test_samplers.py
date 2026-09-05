import numpy as np

from quantum_qubo.problems import random_maxcut
from quantum_qubo.samplers import anneal_then_polish, one_flip_descent, simulated_annealing


def test_simulated_annealing_is_deterministic_for_seed() -> None:
    qubo = random_maxcut(8, 0.5, 4).to_qubo()
    first = simulated_annealing(qubo, num_reads=4, sweeps=30, seed=99)[0]
    second = simulated_annealing(qubo, num_reads=4, sweeps=30, seed=99)[0]
    assert np.array_equal(first.sample, second.sample)
    assert np.isclose(first.energy, second.energy)


def test_one_flip_descent_never_worsens_energy() -> None:
    qubo = random_maxcut(9, 0.5, 5).to_qubo()
    x = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int8)
    polished = one_flip_descent(qubo, x)
    assert polished.energy <= qubo.energy(x) + 1e-12


def test_anneal_then_polish_returns_local_minimum() -> None:
    qubo = random_maxcut(9, 0.45, 8).to_qubo()
    result = anneal_then_polish(qubo, num_reads=6, sweeps=40, seed=11)
    assert min(qubo.flip_delta(result.sample, i) for i in range(qubo.n_variables)) >= -1e-10
