import itertools

import numpy as np

from quantum_qubo.problems import (
    KnapsackProblem,
    MaxCutProblem,
    bounded_binary_weights,
    encode_bounded_integer,
    knapsack_to_qubo,
)
from quantum_qubo.samplers import exact_sample


def test_maxcut_qubo_energy_is_negative_cut() -> None:
    problem = MaxCutProblem(3, ((0, 1, 2.0), (1, 2, 3.0), (0, 2, 1.0)))
    qubo = problem.to_qubo()
    for bits in itertools.product((0, 1), repeat=3):
        x = np.asarray(bits, dtype=np.int8)
        assert np.isclose(qubo.energy(x), -problem.cut_value(x))


def test_bounded_binary_weights_encode_full_range() -> None:
    weights = bounded_binary_weights(17)
    for value in range(18):
        bits = encode_bounded_integer(value, weights)
        assert int(weights @ bits) == value


def test_safe_knapsack_penalty_recovers_true_optimum() -> None:
    problem = KnapsackProblem(
        profits=np.array([8, 6, 5, 9]),
        weights=np.array([4, 3, 2, 5]),
        capacity=7,
    )
    encoded = knapsack_to_qubo(problem)
    result = exact_sample(encoded.qubo)
    decoded = encoded.decode(result.sample)
    brute = max(
        problem.value(np.asarray(bits, dtype=np.int8))
        for bits in itertools.product((0, 1), repeat=problem.n_items)
        if problem.feasible(np.asarray(bits, dtype=np.int8))
    )
    assert decoded["feasible"]
    assert decoded["equality_satisfied"]
    assert decoded["profit"] == brute
    assert np.isclose(result.energy, -float(brute))
