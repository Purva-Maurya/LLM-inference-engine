# TERA — Transformer Engine for Rapid Autoregression
 
A GPT-2 inference engine built entirely from raw weight tensors no `model.generate()`, no `nn.MultiheadAttention`, no high-level shortcuts. Every component (embeddings, attention, sampling, caching, batching, scheduling, memory paging, quantization) is implemented and verified by hand, then benchmarked and stress-tested for real performance and robustness.
 
## Why this project
 
Most ML engineers call `model.generate()` and never see what's inside. This project goes the other direction: understand and rebuild the systems that make LLM inference actually work in production the same techniques behind vLLM, TGI, and every major LLM serving stack. Built incrementally, with every optimization measured and every bug traced to its root cause.
 
## Architecture
 
**Core forward pass:**
```
Prompt → Tokenizer → Embeddings → Transformer block ×12 (attention + MLP + residuals) → Final layernorm → Logits → Sampling → Generated token
```
 
**Serving optimizations layered on top:**
```
KV Cache → Batched Inference → Continuous Batching → Paged KV Cache → Quantization
```
 
**Serving layer:**
```
FastAPI + Docker → live frontend + REST API for real requests
```
 
## What's implemented
 
| # | Feature | File(s) | Result |
|---|---------|---------|--------|
| 1 | Manual GPT-2 forward pass | `model.py` | Verified against HuggingFace reference: max logit diff `5.3e-05`, 100% argmax match |
| 2 | Temperature, top-k, top-p sampling | `sampling.py`, `test_sampling.py` | Coherent, controllable generation |
| 3 | KV caching | `model_cached.py`, `benchmark_kv_cache.py` | **~3x speedup** (representative run: 10.9 → 36.2 tok/s) |
| 4 | Batched inference | `batched_generate.py`, `benchmark_batching.py` | **~1.7–2.5x speedup** over sequential generation |
| 5 | Continuous batching scheduler | `continuous_batching.py` | Dynamic request admission/eviction, no idle compute waiting on the slowest sequence |
| 6 | Paged KV cache | `paged_cache.py`, `paged_generate.py` | Block-based memory allocation (PagedAttention-style), verified correct end-to-end |
| 7 | INT8 quantization (per-tensor + per-channel) | `quantize.py` | **4x memory reduction**; quality tradeoffs measured via perplexity (below) |
| 8 | FastAPI service + Docker deployment | `server.py`, `static/index.html`, `Dockerfile` | Live REST API and frontend, fully containerized |
| 9 | Randomized scheduler stress test | `test_scheduler_stress.py` | 15 randomized variable-length requests, zero cross-request corruption |
| 10 | Paged cache memory accounting | `test_memory_accounting.py` | **2.24x memory reduction (55.4% savings)** vs. naive static allocation, 30-request workload |
| 11 | TTFT latency percentiles | `test_latency_percentiles.py` | Steady-state p50/p95/p99 latency under simulated realistic arrivals |
| 12 | Quantization quality via perplexity | `test_quantization_perplexity.py` | FP32 vs. INT8 per-tensor vs. per-channel, evaluated on a fixed corpus |
 
Every number above can be reproduced with one command:
```bash
python run_all_benchmarks.py
```
*Timing numbers vary somewhat run to run depending on system load; representative values are shown.*
 
## Setup
 
```bash
pip install -r requirements.txt
```
 
First run downloads GPT-2 weights (~500MB) from HuggingFace, cached after.
 
### Run the engine directly
 
```bash
python model.py                        # Verify hand-built forward pass matches HF reference
python test_sampling.py                 # Sampling strategies
python benchmark_kv_cache.py            # KV cache correctness + speed
python benchmark_batching.py            # Batched inference correctness + speed
python continuous_batching.py           # Continuous batching scheduler demo
python paged_generate.py                # Paged KV cache demo
python run_all_benchmarks.py            # Full benchmark + robustness suite
```
 
### Run as a service
 
```bash
uvicorn server:app --reload
# or, containerized:
docker build -t tera .
docker run -p 8000:8000 tera
```
Then open `http://127.0.0.1:8000/` for the live frontend, or `http://127.0.0.1:8000/docs` for the interactive API docs.
 
## Robustness & production-readiness testing
 
Beyond correctness and basic benchmarking, the following verify behavior under more realistic, adversarial conditions:
 
**Randomized scheduler stress test** — 15 requests with randomized prompt lengths (1–8 words) and randomized token budgets (3–15), run through the continuous batching scheduler at `max_batch_size=4`. Verified zero cross-request corruption: every output correctly traces back to its own prompt despite constant admission/eviction churn.
 
**Paged cache memory accounting** — simulated 30 concurrent requests with varying actual lengths (5–63 tokens, avg 27.1) against a 64-token worst-case reservation. Paged allocation used 214 blocks vs. 480 for naive static allocation — a **2.24x memory reduction (55.4% savings)**.
 
**TTFT latency percentiles under simulated realistic arrivals** — 25 requests with randomized prompts and Poisson-distributed arrival timing. Steady-state (post-warm-up): p50 ≈ 24–38ms, p95 ≈ 44–52ms, p99 ≈ 58–64ms (varies by run). An initial run without warm-up showed p99 spike to ~935ms on a single cold-start request — isolated and confirmed as a one-time PyTorch first-call cost, not a recurring tail-latency issue, by adding an unmeasured warm-up call before timing.
 
**Quantization evaluated via perplexity, not just observed text quality** — perplexity measures the model's next-token prediction quality on a fixed evaluation corpus, not general language understanding. Measured on a 297-token fixed corpus:
 
| Configuration | Perplexity | Δ vs FP32 | Weight memory |
|---|---:|---:|---:|
| FP32 (baseline) | 34.719 | — | 324.0 MB |
| INT8 per-tensor | 36.797 | +6.0% | 81.0 MB |
| INT8 per-channel | 33.412 | −3.8% | 81.0 MB |
 
Both quantized configurations achieve a 4x weight-memory reduction. Per-channel quantization showed lower perplexity than per-tensor, and even slightly lower than the FP32 baseline itself on this specific corpus — plausibly within the noise range for a sample this size rather than a genuine general improvement. The FP32 baseline was verified numerically unmutated after both quantization runs (sanity-checked programmatically), ruling out in-place mutation as a confound. *A larger held-out corpus would give a more robust estimate.*
 
## Key implementation notes
 
- **GPT-2's `Conv1D` weight convention** — attention/MLP weights are stored `[in_features, out_features]`, the opposite of `nn.Linear`. Computed as `x @ weight`, never `x @ weight.T`.
- **Causal masking with KV cache** — masking must account for *absolute* sequence position across prefill and decode, not just position within the current chunk. An early bug here caused degenerate output until traced and fixed.
- **Batched padding correctness** — left-padding requires position IDs derived from the attention mask (`cumsum - 1`, clamped), not raw index. Combining causal + padding masks can also fully mask a row, producing `NaN` in softmax that silently corrupts the whole batch via residual connections — fixed by guaranteeing every position can attend to at least itself.
- **Paged memory** — KV cache split into fixed-size blocks; any request uses any available block via a per-request block table, avoiding both over-allocation and fragmentation, mirroring OS virtual memory design.
- **Greedy decoding's repetition trap** — the deployed API initially used greedy decoding, which looped on prompts like "Hello everyone" ("I'm sorry for the delay. I'm sorry for the delay..."). Fixed by wiring temperature + top-p sampling into the serving layer.
- **Free-tier deployment memory limits** — a 512MB-RAM host OOM-killed the container before the server even started. Switched to the CPU-only PyTorch build (no bundled CUDA libraries) to cut the image's memory footprint.

## What's next
 
- Test quantized models with a larger held-out perplexity corpus for a more robust estimate
- Speculative decoding with a small draft model
- Wire the paged cache into the continuous batching scheduler for a unified, production-style serving loop

## Background
 
Built as a systems-focused ML portfolio project — embeddings → attention → sampling → caching → batching → scheduling → memory management → quantization → deployment, one debugged, benchmarked, and stress-tested layer at a time.
 