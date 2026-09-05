from quantum_qubo.benchmark import benchmark_knapsack, benchmark_maxcut


def test_benchmark_smoke_has_valid_metrics() -> None:
    maxcut = benchmark_maxcut(instances=2, n_nodes=8, seed=3, num_reads=4, sweeps=30)
    knapsack = benchmark_knapsack(instances=2, n_items=5, seed=7, num_reads=4, sweeps=30)
    for report in (maxcut, knapsack):
        for _key, value in report.items():
            if isinstance(value, dict) and "feasibility_rate" in value:
                assert 0.0 <= value["exact_hit_rate"] <= 1.0
                assert 0.0 <= value["feasibility_rate"] <= 1.0
                assert value["mean_relative_gap"] >= 0.0
