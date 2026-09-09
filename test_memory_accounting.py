import random
import math
from paged_cache import BlockPool, PagedRequest

# GPT-2 config, matching your model
N_LAYER = 12
N_HEADS = 12
HEAD_DIM = 64
BYTES_PER_VALUE = 4  # fp32

def bytes_for_blocks(num_blocks, block_size):
    """Memory footprint of a given number of KV cache blocks, across all layers, K and V."""
    values_per_block = block_size * N_HEADS * HEAD_DIM
    bytes_per_block = values_per_block * BYTES_PER_VALUE
    total_bytes = num_blocks * bytes_per_block * N_LAYER * 2  # ×2 for K and V
    return total_bytes


def run_memory_comparison():
    random.seed(7)

    num_requests = 30
    max_possible_len = 64       # worst-case length the naive approach must reserve for
    block_size = 4

    # Simulate realistic, varying actual lengths (most requests are much shorter than worst-case)
    actual_lengths = [random.randint(3, max_possible_len) for _ in range(num_requests)]

    print(f"Simulating {num_requests} concurrent requests")
    print(f"Actual lengths (tokens): min={min(actual_lengths)}, max={max(actual_lengths)}, "
          f"avg={sum(actual_lengths)/len(actual_lengths):.1f}")
    print(f"Naive worst-case reservation per request: {max_possible_len} tokens\n")

    # --- Naive approach: every request reserves the full worst-case buffer ---
    naive_blocks_per_request = math.ceil(max_possible_len / block_size)
    naive_total_blocks = naive_blocks_per_request * num_requests
    naive_bytes = bytes_for_blocks(naive_total_blocks, block_size)

    # --- Paged approach: allocate real blocks via your actual BlockPool/PagedRequest code ---
    pool = BlockPool(num_blocks=naive_total_blocks, block_size=block_size,
                      n_layer=N_LAYER, n_heads=N_HEADS, head_dim=HEAD_DIM)

    requests = [PagedRequest(request_id=i, pool=pool) for i in range(num_requests)]

    for req, length in zip(requests, actual_lengths):
        for _ in range(length):
            req.append_token_slot()

    paged_blocks_used = sum(len(req.block_table) for req in requests)
    paged_bytes = bytes_for_blocks(paged_blocks_used, block_size)

    # --- Report ---
    print(f"NAIVE   — blocks reserved: {naive_total_blocks:4d}  memory: {naive_bytes / 1024:.1f} KB")
    print(f"PAGED   — blocks actually used: {paged_blocks_used:4d}  memory: {paged_bytes / 1024:.1f} KB")

    savings_pct = (1 - paged_bytes / naive_bytes) * 100
    print(f"\nMemory saved by paging: {savings_pct:.1f}%")
    print(f"Reduction factor: {naive_bytes / paged_bytes:.2f}x")

    for req in requests:
        req.free()


if __name__ == "__main__":
    run_memory_comparison()