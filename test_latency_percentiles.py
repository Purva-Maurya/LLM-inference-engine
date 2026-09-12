import time
import random
import numpy as np

from model_cached import generate_cached


def generate_random_prompts(n, vocab, min_words=1, max_words=8):
    prompts = []

    for _ in range(n):
        length = random.randint(min_words, max_words)
        prompts.append(" ".join(random.choices(vocab, k=length)))

    return prompts


def measure_ttft(prompt):
    """Time to first token: run prefill + one decode step, time just that."""
    start = time.perf_counter()

    # max_new_tokens=1 isolates prefill + first token generation
    generate_cached(prompt, max_new_tokens=1)

    ttft = time.perf_counter() - start
    return ttft


def run_latency_test():
    random.seed(11)

    vocab = [
        "the", "cat", "runs", "fast", "quantum", "engine",
        "hello", "world", "future", "code", "model",
        "data", "system", "test"
    ]

    num_requests = 25
    prompts = generate_random_prompts(num_requests, vocab)

    ttfts = []

    print(
        f"Submitting {num_requests} requests "
        f"with simulated realistic arrival delays...\n"
    )

    # Warm-up call — NOT measured
    # Separates cold-start cost from steady-state latency
    _ = generate_cached("warm up", max_new_tokens=1)

    print("(warm-up complete, starting timed run)\n")

    # IMPORTANT: this loop must be INSIDE run_latency_test()
    for i, prompt in enumerate(prompts):

        # Simulate realistic, uneven arrival timing
        delay = random.expovariate(3.0)
        time.sleep(min(delay, 1.0))

        ttft = measure_ttft(prompt)
        ttfts.append(ttft)

        print(
            f"  Request {i:2d}: "
            f"prompt={prompt!r:40s} "
            f"TTFT={ttft * 1000:.1f}ms"
        )

    # Convert seconds → milliseconds
    ttfts_ms = np.array(ttfts) * 1000

    p50 = np.percentile(ttfts_ms, 50)
    p95 = np.percentile(ttfts_ms, 95)
    p99 = np.percentile(ttfts_ms, 99)

    print("\n=== TTFT Latency Distribution (ms) ===")
    print(f"  Min:  {ttfts_ms.min():.1f}")
    print(f"  p50:  {p50:.1f}")
    print(f"  p95:  {p95:.1f}")
    print(f"  p99:  {p99:.1f}")
    print(f"  Max:  {ttfts_ms.max():.1f}")


if __name__ == "__main__":
    run_latency_test()