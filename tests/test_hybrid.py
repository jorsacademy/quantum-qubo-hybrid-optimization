import numpy as np

from quantum_qubo.hybrid import repair_knapsack_sample
from quantum_qubo.problems import KnapsackProblem, knapsack_to_qubo


def test_knapsack_repair_is_feasible_and_recomputes_slack() -> None:
    problem = KnapsackProblem(
        profits=np.array([12, 10, 8, 7]),
        weights=np.array([6, 5, 4, 3]),
        capacity=9,
    )
    encoded = knapsack_to_qubo(problem)
    sample = np.ones(encoded.qubo.n_variables, dtype=np.int8)
    repaired = repair_knapsack_sample(encoded, sample)
    decoded = encoded.decode(repaired)
    assert decoded["feasible"]
    assert decoded["equality_satisfied"]
