import random
from continuous_batching import Request, run_scheduler

def generate_random_prompts(n, vocab, min_words=1, max_words=8):
    """Build n prompts of random length from a small word pool, so tokenized
    lengths vary naturally across requests."""
    prompts = []
    for _ in range(n):
        length = random.randint(min_words, max_words)
        words = random.choices(vocab, k=length)
        prompts.append(" ".join(words))
    return prompts


def test_randomized_scheduler():
    random.seed(42)  # reproducible run — same "random" test every time

    vocab = ["the", "cat", "runs", "fast", "quantum", "engine", "hello",
             "world", "future", "code", "model", "data", "system", "test"]

    num_requests = 15
    prompts = generate_random_prompts(num_requests, vocab)
    max_tokens_list = [random.randint(3, 15) for _ in range(num_requests)]

    print(f"Running {num_requests} randomized requests through scheduler...")
    for i, (p, m) in enumerate(zip(prompts, max_tokens_list)):
        print(f"  Request {i}: prompt={p!r} max_new_tokens={m}")

    outputs = run_scheduler(prompts, max_tokens_list, max_batch_size=4)

    # --- Correctness checks ---
    assert len(outputs) == num_requests, \
        f"Expected {num_requests} outputs, got {len(outputs)}"

    failures = []
    for i, (prompt, output) in enumerate(zip(prompts, outputs)):
        if not output.startswith(prompt):
            failures.append((i, prompt, output))

    if failures:
        print(f"\nWRONG {len(failures)} request(s) show cross-contamination or corruption:")
        for i, prompt, output in failures:
            print(f"  Request {i}: expected prefix {prompt!r}, got {output!r}")
    else:
        print(f"\nRIGHT All {num_requests} requests completed correctly — each output starts with its own prompt.")

    assert not failures, "Some requests' outputs don't match their own prompts — likely cross-request corruption"


if __name__ == "__main__":
    test_randomized_scheduler()