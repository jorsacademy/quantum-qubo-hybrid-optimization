# Research Notes

## Scope

This repository evaluates QUBO formulation and hybrid sampling workflows. It does not claim quantum advantage.

## Why Max-Cut and knapsack

Weighted Max-Cut provides a clean unconstrained benchmark: the binary quadratic energy can be algebraically checked against the original combinatorial objective without a penalty parameter.

Knapsack provides the opposite case: a constrained binary optimization problem must be mapped into an unconstrained QUBO. It exposes penalty selection, slack representation, infeasible low-energy samples, and repair semantics.

## Penalty derivation

For integer knapsack, define

`E(x,s) = -p·x + P(w·x + s - C)^2`.

If `P > sum(p)`, then any binary state violating the equality has residual magnitude at least one and

`E_infeasible >= -sum(p) + P > 0`.

The all-zero item vector with `s=C` is feasible and has energy zero. Therefore an infeasible state cannot be a global minimizer. The exact ground-state test enforces this argument in code.

This does not mean arbitrary heuristic samples are feasible. A heuristic can stop at an excited state, which is why raw-SA feasibility is reported separately.

## Hybrid post-processing semantics

The repository distinguishes three outputs:

1. raw simulated-annealing sample;
2. same QUBO sample after generic 1-flip descent;
3. knapsack candidate after problem-aware feasibility repair and slack recomputation.

Only (1) is the direct sampler output. Reporting (2) or (3) as if it were raw quantum/annealing performance would be misleading.

## Quantum execution

`sample_qpu` uses `EmbeddingComposite(DWaveSampler())`. Logical-to-physical embedding may introduce chains, and hardware studies should report chain strength, broken-chain handling, annealing schedule, reads and solver identity.

`sample_leap_hybrid` uses `LeapHybridSampler`, which is a quantum-classical hybrid service. Results from it should be labeled hybrid rather than pure quantum annealing.

## Benchmarking rules

A future hardware benchmark should record at least:

- exact instance-generation seed and QUBO coefficients;
- logical variable and interaction counts;
- sampler/solver identifier;
- reads and annealing or hybrid time limit;
- preprocessing and post-processing included in the timing boundary;
- raw and repaired solution quality separately;
- feasibility rate;
- optimality gap against an exact or best-known reference;
- repeated-run uncertainty;
- time-to-target or time-to-solution rather than a single best sample when possible.

## Current CPU smoke interpretation

On six small deterministic instances, classical local polishing closes the Max-Cut gaps left by the intentionally modest simulated-annealing budget. Knapsack is harder: repair restores feasibility but does not generally recover the optimum. This is useful because it prevents the repository from turning hybrid post-processing into an unqualified success claim.

## D-Wave documentation alignment

The implementation follows the current documented modeling concepts:

- BQM/QUBO models are unconstrained binary quadratic objectives;
- constraints can be encoded through penalty models;
- inequalities can be converted to equalities using slack variables;
- `LeapHybridSampler` samples BQMs through the cloud hybrid service;
- direct QPU use requires a sampler and typically a minor-embedding composite.

The optional dependency is intentionally separated from the core package so reproducibility tests do not depend on credentials or a particular cloud solver being available.
