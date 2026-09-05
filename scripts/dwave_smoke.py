from __future__ import annotations

import argparse
import json

from quantum_qubo.dwave_adapter import sample_leap_hybrid, sample_qpu
from quantum_qubo.problems import random_maxcut


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("leap-hybrid", "qpu"), default="leap-hybrid")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nodes", type=int, default=20)
    parser.add_argument("--time-limit", type=int, default=None)
    parser.add_argument("--num-reads", type=int, default=100)
    args = parser.parse_args()

    problem = random_maxcut(args.nodes, 0.3, args.seed)
    qubo = problem.to_qubo()
    if args.backend == "leap-hybrid":
        result = sample_leap_hybrid(qubo, time_limit=args.time_limit)
    else:
        result = sample_qpu(qubo, num_reads=args.num_reads)
    print(
        json.dumps(
            {
                "backend": result.solver,
                "energy": result.energy,
                "cut_value": problem.cut_value(result.sample),
                "sample": result.sample.tolist(),
                "info": result.info,
            },
            indent=2,
            default=str,
        )
    )


if __name__ == "__main__":
    main()
