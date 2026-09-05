from __future__ import annotations

import argparse
import json
from pathlib import Path

from quantum_qubo import benchmark_knapsack, benchmark_maxcut


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instances", type=int, default=6)
    parser.add_argument("--reads", type=int, default=24)
    parser.add_argument("--sweeps", type=int, default=220)
    parser.add_argument("--output", type=Path, default=Path("artifacts/benchmark.json"))
    args = parser.parse_args()

    report = {
        "maxcut": benchmark_maxcut(
            instances=args.instances,
            seed=args.seed + 1000,
            num_reads=args.reads,
            sweeps=args.sweeps,
        ),
        "knapsack": benchmark_knapsack(
            instances=args.instances,
            seed=args.seed + 2000,
            num_reads=args.reads,
            sweeps=args.sweeps,
        ),
        "policy": {
            "quantum_speedup_claimed": False,
            "reason": "No QPU/Leap timing is produced by CPU CI.",
        },
        "config": vars(args) | {"output": str(args.output)},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
