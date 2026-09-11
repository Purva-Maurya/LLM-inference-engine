
# TERA - Transformer Engine for Rapid Autoregression. 



A GPT-2 inference engine built entirely from raw weight tensors — no `model.generate()`, no `nn.MultiheadAttention`, no high-level shortcuts. Every component (embeddings, attention, sampling, caching, batching, memory paging) is implemented and verified by hand, then benchmarked for real performance gains.

Built incrementally over six weeks, each with working code, debugged bugs, and measured results.

## Why this project

Most ML engineers call `model.generate()` and never see what's inside. This project goes the other direction: understand and rebuild the systems that make LLM inference actually work in production — the same techniques behind vLLM, TGI, and every major LLM serving stack.

## What's implemented

| Week | Feature | File(s) | Result |
|------|---------|---------|--------|
| 1 | Manual GPT-2 forward pass | `model.py` | Verified against HuggingFace reference: max logit diff `5.3e-05`, 100% argmax match |
| 2 | Temperature, top-k, top-p sampling | `sampling.py`, `test_sampling.py` | Coherent, controllable generation |
| 3 | KV caching | `model_cached.py`, `benchmark_kv_cache.py` | **3.24x speedup** (18.2 → 59.0 tokens/sec) |
| 4 | Batched inference | `batched_generate.py`, `benchmark_batching.py` | **2.55x speedup** over sequential generation |
| 5 | Continuous batching scheduler | `continuous_batching.py` | Dynamic request admission/eviction, no idle compute waiting on the slowest sequence |
| 6 | Paged KV cache | `paged_cache.py`, `paged_generate.py` | Block-based memory allocation (PagedAttention-style), verified correct end-to-end |
| 7 | INT8 quantization (per-tensor + per-channel) | `quantize.py` | **4x memory reduction**; per-tensor quantization degraded generation quality, per-channel measurably improved fidelity but didn't fully eliminate it |
| 8| Paged KV cache memory savings Simulated 30 concurrent requests with varying actual lengths (5–63 tokens, avg. 27.1) against a 64-token worst-case reservation. TERA's paged cache used 214 blocks vs. 480 blocks for naive static allocation, resulting in a **55.4% reduction** in KV-cache memory usage (2.24× less memory).


## Architecture

Layered on top of this core pipeline: KV caching (avoid recomputation), batching (process multiple requests together), continuous batching (dynamic scheduling), and paged memory (efficient, fragmentation-free KV storage).

## Setup

```bash
pip install torch transformers
```

First run downloads GPT-2 weights (~500MB) from HuggingFace, cached after.

## Run it

```bash
# Verify the hand-built forward pass matches the reference implementation
python model.py

# Sampling strategies
python test_sampling.py

# KV cache correctness + speed
python benchmark_kv_cache.py

# Batched inference correctness + speed
python benchmark_batching.py

# Continuous batching scheduler demo
python continuous_batching.py

# Paged KV cache demo
python paged_generate.py
```

## Key implementation notes

- **GPT-2's `Conv1D` weight convention**: attention/MLP weights are stored `[in_features, out_features]`, the opposite of `nn.Linear` — computed as `x @ weight`, never `x @ weight.T`.
- **Causal masking with KV cache**: masking must account for *absolute* sequence position (prefill vs. decode), not just position within the current chunk — an early bug here caused degenerate output until fixed.
- **Batched padding**: left-padding requires position IDs derived from the attention mask (`cumsum - 1`, clamped), not raw index — otherwise padded sequences get corrupted positional information.
- **All-masked-row NaN**: combining causal + padding masks can fully mask a row (softmax → `NaN`), which silently corrupts the whole batch through residual connections. Fixed by guaranteeing every position can attend to itself.
- **Paged memory**: KV cache is split into fixed-size blocks; any request can use any available block via a per-request block table, avoiding both over-allocation and fragmentation — the same idea behind OS virtual memory paging.

## Week 7 findings: quantization tradeoffs

Implemented INT8 weight quantization at two granularities and compared generation quality against the fp32 baseline:

- **Per-tensor quantization**: 4x memory reduction, small per-weight reconstruction error (mean abs error ~0.009), but generation diverged into a different repetitive pattern than the original.
- **Per-channel quantization**: same 4x memory reduction, measurably better fidelity — exactly reproduced the original model's opening sentence — but still fell into its own repetition loop after that.

This suggests that while finer-grained quantization reduces error, the residual error is still large enough to affect generation dynamics on a small (124M parameter) model, particularly under greedy decoding. Worth testing with sampling instead of greedy decoding, and/or on a larger model, as a next step.

## What's next

- Test quantized models with temperature/top-p sampling instead of greedy decoding
- Speculative decoding with a draft model
- Wiring paged cache into the continuous batching scheduler for full production-style serving

## Background

Built as a systems-focused ML portfolio project, going through embeddings → attention → sampling → caching → batching → memory management, one debugged and benchmarked layer at a time.