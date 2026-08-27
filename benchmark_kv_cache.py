import time
import torch
from model import tokenizer, greedy_generate
from model_cached import generate_cached

prompt = "The future of artificial intelligence is"
max_new_tokens = 30

# --- Uncached (Week 1 version) ---
start = time.perf_counter()
output_uncached = greedy_generate(prompt, max_new_tokens=max_new_tokens)
uncached_time = time.perf_counter() - start

# --- Cached (Week 3 version) ---
start = time.perf_counter()
output_cached = generate_cached(prompt, max_new_tokens=max_new_tokens)
cached_time = time.perf_counter() - start

print(f"Uncached time: {uncached_time:.3f}s  ({max_new_tokens/uncached_time:.1f} tokens/sec)")
print(f"Cached time:   {cached_time:.3f}s  ({max_new_tokens/cached_time:.1f} tokens/sec)")
print(f"Speedup: {uncached_time/cached_time:.2f}x")

assert output_uncached == output_cached, "Outputs differ! Something's wrong."
print("\nOutputs match — cache is numerically correct.")