import numpy as np

from quantum_qubo.model import QUBO


def test_upper_triangular_matrix_matches_energy() -> None:
    qubo = QUBO(
        linear=np.array([-1.0, 2.0, 0.5]),
        quadratic=np.array([[0.0, 3.0, -2.0], [3.0, 0.0, 1.0], [-2.0, 1.0, 0.0]]),
        offset=4.0,
    )
    x = np.array([1, 0, 1], dtype=np.int8)
    assert np.isclose(qubo.energy(x), 4.0 + x @ qubo.upper_triangular_matrix() @ x)


def test_flip_delta_matches_recomputed_energy() -> None:
    qubo = QUBO(
        linear=np.array([-2.0, 1.0]),
        quadratic=np.array([[0.0, 4.0], [4.0, 0.0]]),
    )
    x = np.array([1, 1], dtype=np.int8)
    before = qubo.energy(x)
    delta = qubo.flip_delta(x, 0)
    y = x.copy()
    y[0] = 0
    assert np.isclose(before + delta, qubo.energy(y))
