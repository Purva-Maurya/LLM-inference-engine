"""
Runs every benchmark in the TERA project and prints a consolidated report.

Each benchmark is executed as a separate subprocess so that model state
does not leak between experiments.
"""

import subprocess
import sys


BENCHMARKS = [
    ("KV caching speedup", "benchmark_kv_cache.py"),
    ("Batched inference speedup", "benchmark_batching.py"),
    ("Scheduler stress test", "test_scheduler_stress.py"),
    ("Paged cache memory accounting", "test_memory_accounting.py"),
    ("TTFT latency percentiles", "test_latency_percentiles.py"),
    ("Quantization perplexity", "test_quantization_perplexity.py"),
]


def run_benchmark(name, script):
    print("\n" + "=" * 70)
    print(f"  {name}")
    print(f"  Script: {script}")
    print("=" * 70)

    result = subprocess.run(
        [sys.executable, script],
        capture_output=False
    )

    if result.returncode == 0:
        print(f"\n✅ {script} completed successfully.")
        return True

    print(f"\n❌ {script} failed with exit code {result.returncode}.")
    return False


def main():
    print("\n" + "=" * 70)
    print("  TERA — FULL BENCHMARK SUITE")
    print("=" * 70)

    passed = 0
    failed = 0

    for name, script in BENCHMARKS:
        if run_benchmark(name, script):
            passed += 1
        else:
            failed += 1

    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Passed: {passed}")
    print(f"  Failed: {failed}")

    if failed == 0:
        print("\n  ✅ All benchmarks completed successfully.")
    else:
        print("\n  ⚠️ Some benchmarks failed. Check the output above.")

    print("=" * 70)


if __name__ == "__main__":
    main()