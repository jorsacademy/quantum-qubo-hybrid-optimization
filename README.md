# Quantum QUBO Hybrid Optimization

A reproducible research implementation for **Quadratic Unconstrained Binary Optimization (QUBO)** with exact classical validation, simulated annealing, classical hybrid post-processing, and optional D-Wave Leap Hybrid / QPU execution.

The repository is deliberately structured so that ordinary CPU GitHub Actions can validate the mathematics without pretending to be a quantum benchmark. A quantum or quantum-classical speedup is never inferred from CPU-only experiments.

## What is implemented

### 1. Weighted Max-Cut → QUBO

For binary side indicators `x_i ∈ {0,1}`, an edge `(i,j)` of weight `w_ij` contributes

```text
w_ij (x_i + x_j - 2 x_i x_j)
```

to the cut. Because QUBO samplers minimize energy, the implementation minimizes the negative cut value. Tests exhaustively verify on a small graph that

```text
QUBO energy == -weighted cut value
```

for every binary assignment.

### 2. 0-1 knapsack → QUBO with binary slack

The capacity inequality

```text
weights · x <= capacity
```

is converted to

```text
weights · x + slack = capacity
```

using bounded binary slack bits. The QUBO objective is

```text
-profit · x + P (weights · x + slack - capacity)^2
```

with `P > sum(profits)`.

For positive integer weights/profits, this is a sufficient penalty bound for the implementation: any equality-violating assignment incurs at least `P`, while its best possible profit contribution is bounded below by `-sum(profits)`. Since a feasible all-zero item assignment exists, a global QUBO minimizer must satisfy the capacity equality.

The test suite confirms that the exact QUBO ground state recovers the true brute-force knapsack optimum.

## Solvers and samplers

### Exact enumeration

`exact_sample()` enumerates all binary assignments for small QUBOs. It is not intended to scale; it exists as an independent correctness oracle for CI-sized problems.

### Simulated annealing

The repository includes a deterministic, NumPy-only Metropolis simulated annealer with an energy-scale-aware inverse-temperature schedule. It does not require Ocean SDK, a cloud account, or quantum hardware.

### Classical hybrid polishing

`anneal_then_polish()` takes low-energy annealing samples and applies deterministic 1-flip descent. For knapsack, `repair_knapsack_sample()` additionally projects an arbitrary candidate to a feasible item selection and recomputes exact slack bits.

This is intentionally called **hybrid post-processing**, not quantum advantage: it combines a global stochastic sampler with deterministic classical local improvement/repair.

### Optional D-Wave execution

Install the optional integration:

```bash
python -m pip install -e ".[dwave]"
```

Then use either:

```bash
python scripts/dwave_smoke.py --backend leap-hybrid
```

or, with configured QPU access:

```bash
python scripts/dwave_smoke.py --backend qpu --num-reads 100
```

The adapter uses Ocean's `BinaryQuadraticModel` representation and supports:

- `LeapHybridSampler` for D-Wave's cloud quantum-classical hybrid BQM solver
- `DWaveSampler` wrapped by `EmbeddingComposite` for direct QPU sampling

Credentials and solver selection are left to normal D-Wave/Ocean configuration. They are never committed to this repository.

## Why the architecture separates CPU CI from D-Wave execution

D-Wave's Ocean model treats QUBOs as binary quadratic models sampled by interchangeable samplers. This makes it possible to validate the same QUBO coefficients with exact enumeration and classical annealing before submitting them to a cloud hybrid solver or QPU.

GitHub-hosted runners do not provide D-Wave credentials or quantum hardware. Therefore CI checks:

- QUBO energy identities,
- exact ground states,
- penalty correctness,
- deterministic simulated annealing,
- local-search monotonicity,
- knapsack feasibility repair,
- benchmark/report generation.

It does **not** report quantum wall-clock speedup, time-to-solution advantage, or quantum advantage.

## Reproducible benchmark

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest
python scripts/run_benchmark.py \
  --seed 42 \
  --instances 6 \
  --reads 24 \
  --sweeps 220 \
  --output artifacts/benchmark.json
```

### Current local smoke result

The deterministic CPU configuration above produced:

| Problem / method | Mean gap | Max gap | Exact-hit rate | Feasibility |
|---|---:|---:|---:|---:|
| Max-Cut — simulated annealing | 3.91% | 6.72% | 16.7% | 100% |
| Max-Cut — SA + 1-flip polish | **0.00%** | **0.00%** | **100%** | 100% |
| Knapsack — simulated annealing | 53.32% | 100% | 0% | 83.3% |
| Knapsack — SA + polish + repair | **11.47%** | 24.42% | 16.7% | **100%** |

These are **small synthetic smoke experiments**, not evidence of quantum advantage. They show two more limited conclusions:

1. deterministic local polishing can materially improve low-energy Max-Cut samples;
2. problem-aware repair can restore knapsack feasibility, but does not guarantee optimality.

GitHub Actions reruns the same benchmark and stores the JSON artifact.

## D-Wave / QUBO methodological notes

D-Wave documentation describes QUBOs/BQMs as unconstrained quadratic binary models; constraints are commonly represented using penalty terms. Linear inequalities can be converted to equalities by introducing nonnegative slack variables, which is the construction used here.

D-Wave's current Ocean documentation also exposes `LeapHybridSampler.sample(bqm, ...)` for BQM hybrid solving. Direct QPU sampling requires embedding the logical QUBO graph into the hardware graph; Ocean's `EmbeddingComposite(DWaveSampler())` handles that mapping layer.

The included pure-Python simulated annealer is used for reproducibility. If Ocean is installed, D-Wave's `dwave-samplers` package provides its own production implementation of simulated annealing, but that package is not needed by the core CI.

## Repository layout

```text
src/quantum_qubo/
  model.py          QUBO representation and energy/delta calculations
  problems.py       Max-Cut and knapsack formulations
  samplers.py       exact enumeration, simulated annealing, 1-flip polish
  hybrid.py         problem-aware knapsack repair
  dwave_adapter.py  optional Leap Hybrid / QPU adapter
  benchmark.py      exact-reference benchmark metrics
scripts/
  run_benchmark.py
  dwave_smoke.py
tests/
.github/workflows/ci.yml
```

## Tests

The suite checks:

- dense/upper-triangular QUBO energy consistency;
- flip-delta correctness;
- Max-Cut energy equivalence for every state of a tiny graph;
- bounded binary slack encoding over its full range;
- safe-penalty knapsack ground state versus direct brute force;
- deterministic simulated annealing;
- monotonic 1-flip local descent;
- local-minimum termination;
- knapsack repair feasibility and equality consistency;
- benchmark metric validity.

## Research limitations

- Exact enumeration is exponential and used only for small validation instances.
- Simulated annealing quality depends on reads, sweeps and temperature schedule.
- The knapsack penalty guarantee assumes positive integer weights/profits and unit-minimum equality violation.
- Classical repair changes the sampled state; repaired performance must be distinguished from raw sampler performance.
- QPU experiments require embedding, chain-strength choices, gauge choices, reads, anneal schedules and hardware calibration context.
- Leap Hybrid is a quantum-classical cloud service; its result should not be described as a pure-QPU result.
- A fair quantum/classical comparison needs repeated runs, time-to-solution definitions, preprocessing/accounting rules and matched solution-quality targets.

## Primary references

- Glover, Kochenberger & Du (2019), *A Tutorial on Formulating and Using QUBO Models*.
- Kirkpatrick, Gelatt & Vecchi (1983), *Optimization by Simulated Annealing*.
- D-Wave Ocean SDK documentation: Binary Quadratic Models, QUBO formulation and sampling workflow.
- D-Wave documentation: reformulating linear inequalities with slack variables and penalty functions.
- D-Wave Ocean API: `LeapHybridSampler.sample`.
- D-Wave Ocean documentation: Max-Cut and direct QPU sampling examples.

See `RESEARCH_NOTES.md` for additional methodology and interpretation guidance.

## License

MIT. D-Wave Ocean and D-Wave cloud services are separate products with their own licenses, terms and system requirements.
