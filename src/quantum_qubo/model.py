from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class QUBO:
    """Binary quadratic model using one coefficient per unordered pair.

    Energy is
        offset + linear @ x + sum_{i<j} quadratic[i,j] * x[i] * x[j].
    `quadratic` is stored symmetrically with a zero diagonal for convenient local fields.
    """

    linear: np.ndarray
    quadratic: np.ndarray
    offset: float = 0.0
    labels: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        linear = np.asarray(self.linear, dtype=float)
        quadratic = np.asarray(self.quadratic, dtype=float)
        if linear.ndim != 1:
            raise ValueError("linear must be one-dimensional")
        n = linear.size
        if quadratic.shape != (n, n):
            raise ValueError("quadratic must have shape (n, n)")
        if not np.allclose(quadratic, quadratic.T):
            raise ValueError("quadratic must be symmetric")
        if not np.allclose(np.diag(quadratic), 0.0):
            raise ValueError("quadratic diagonal must be zero")
        labels = self.labels or tuple(f"x{i}" for i in range(n))
        if len(labels) != n:
            raise ValueError("labels length must match number of variables")
        object.__setattr__(self, "linear", linear)
        object.__setattr__(self, "quadratic", quadratic)
        object.__setattr__(self, "labels", tuple(labels))

    @property
    def n_variables(self) -> int:
        return int(self.linear.size)

    def validate_sample(self, sample: np.ndarray) -> np.ndarray:
        x = np.asarray(sample, dtype=np.int8)
        if x.shape != (self.n_variables,):
            raise ValueError("sample has wrong shape")
        if np.any((x != 0) & (x != 1)):
            raise ValueError("QUBO samples must be binary")
        return x

    def energy(self, sample: np.ndarray) -> float:
        x = self.validate_sample(sample).astype(float)
        pair_term = 0.5 * float(x @ self.quadratic @ x)
        return float(self.offset + self.linear @ x + pair_term)

    def flip_delta(self, sample: np.ndarray, index: int) -> float:
        x = self.validate_sample(sample)
        if not 0 <= index < self.n_variables:
            raise IndexError(index)
        field = float(self.linear[index] + self.quadratic[index] @ x)
        return float((1 - 2 * int(x[index])) * field)

    def upper_triangular_matrix(self) -> np.ndarray:
        """Return a matrix Q such that x.T @ Q @ x equals this model's non-offset energy."""
        q = np.zeros((self.n_variables, self.n_variables), dtype=float)
        np.fill_diagonal(q, self.linear)
        q += np.triu(self.quadratic, k=1)
        return q

    def to_dict(self) -> dict[str, object]:
        q = self.upper_triangular_matrix()
        return {
            "labels": list(self.labels),
            "linear": self.linear.tolist(),
            "quadratic_upper": q.tolist(),
            "offset": float(self.offset),
        }
