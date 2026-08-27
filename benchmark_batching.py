import time
from model import greedy_generate
from batched_generate import generate_batched

prompts = [
    "Hello",
    "The weather today is",
    "I really enjoy programming because"
]
max_new_tokens = 20

# --- Sequential: one prompt at a time ---
start = time.perf_counter()
sequential_outputs = [greedy_generate(p, max_new_tokens=max_new_tokens) for p in prompts]
sequential_time = time.perf_counter() - start

# --- Batched: all prompts together ---
start = time.perf_counter()
batched_outputs = generate_batched(prompts, max_new_tokens=max_new_tokens)
batched_time = time.perf_counter() - start

total_tokens = len(prompts) * max_new_tokens

print(f"Sequential time: {sequential_time:.3f}s  ({total_tokens/sequential_time:.1f} tokens/sec)")
print(f"Batched time:    {batched_time:.3f}s  ({total_tokens/batched_time:.1f} tokens/sec)")
print(f"Speedup: {sequential_time/batched_time:.2f}x")