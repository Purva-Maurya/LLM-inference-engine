import torch

class BlockPool:
    """
    Manages a fixed pool of KV cache blocks, shared across all requests.
    Each block holds K and V for `block_size` token positions, across all
    layers, all heads.
    """
    def __init__(self, num_blocks, block_size, n_layer=12, n_heads=12, head_dim=64):
        self.block_size = block_size
        self.num_blocks = num_blocks

        # Physical storage: one big tensor per layer, shaped to hold ALL blocks.
        # Shape: [num_blocks, block_size, n_heads, head_dim] per layer, for K and V separately.
        self.k_blocks = [
            torch.zeros(num_blocks, block_size, n_heads, head_dim)
            for _ in range(n_layer)
        ]
        self.v_blocks = [
            torch.zeros(num_blocks, block_size, n_heads, head_dim)
            for _ in range(n_layer)
        ]

        self.free_blocks = list(range(num_blocks))  # which physical block ids are unused

    def allocate_block(self):
        if not self.free_blocks:
            raise RuntimeError("Out of KV cache blocks!")
        return self.free_blocks.pop()

    def free_block(self, block_id):
        self.free_blocks.append(block_id)


if __name__ == "__main__":
    pool = BlockPool(num_blocks=8, block_size=4)
    print(f"Total blocks: {pool.num_blocks}, free: {len(pool.free_blocks)}")

    b1 = pool.allocate_block()
    b2 = pool.allocate_block()
    print(f"Allocated blocks: {b1}, {b2}")
    print(f"Free blocks remaining: {len(pool.free_blocks)}")

    pool.free_block(b1)
    print(f"After freeing one block, free remaining: {len(pool.free_blocks)}")

class PagedRequest:
    """
    Tracks one request's block table: which physical blocks hold its tokens,
    in logical order.
    """
    def __init__(self, request_id, pool: BlockPool):
        self.request_id = request_id
        self.pool = pool
        self.block_table = []       # list of physical block ids, in logical order
        self.num_tokens = 0         # how many real tokens stored so far

    def num_tokens_in_last_block(self):
        if self.num_tokens == 0:
            return 0
        return self.num_tokens % self.pool.block_size or self.pool.block_size

    def needs_new_block(self):
        return self.num_tokens % self.pool.block_size == 0

    def append_token_slot(self):
        """Reserve space for one new token, allocating a new block if needed."""
        if self.needs_new_block():
            new_block = self.pool.allocate_block()
            self.block_table.append(new_block)
        self.num_tokens += 1

    def free(self):
        for block_id in self.block_table:
            self.pool.free_block(block_id)
        self.block_table = []


if __name__ == "__main__":
    pool = BlockPool(num_blocks=8, block_size=4)
    req = PagedRequest(request_id=0, pool=pool)

    # Simulate adding 10 tokens one at a time
    for i in range(10):
        req.append_token_slot()
        print(f"Token {i}: num_tokens={req.num_tokens}, block_table={req.block_table}, free_blocks={len(pool.free_blocks)}")

    req.free()
    print(f"\nAfter freeing request: free_blocks={len(pool.free_blocks)}")


def write_kv(pool: BlockPool, req: PagedRequest, layer_idx, k, v):
    """
    Write one new token's K/V (for one layer) into its assigned physical block slot.
    k, v: shape [n_heads, head_dim] — a single token's K/V for this layer.
    """
    block_id = req.block_table[-1]                      # most recently allocated block
    slot = (req.num_tokens - 1) % pool.block_size        # position within that block

    pool.k_blocks[layer_idx][block_id, slot] = k
    pool.v_blocks[layer_idx][block_id, slot] = v


def gather_kv(pool: BlockPool, req: PagedRequest, layer_idx):
    """
    Reconstruct this request's full K/V sequence (for one layer) by reading
    across all its physical blocks, in logical order, trimmed to actual length.
    """
    k_chunks = [pool.k_blocks[layer_idx][block_id] for block_id in req.block_table]
    v_chunks = [pool.v_blocks[layer_idx][block_id] for block_id in req.block_table]

    k_full = torch.cat(k_chunks, dim=0)   # [num_blocks * block_size, n_heads, head_dim]
    v_full = torch.cat(v_chunks, dim=0)

    return k_full[:req.num_tokens], v_full[:req.num_tokens]  # trim padding beyond real tokens


if __name__ == "__main__":
    pool = BlockPool(num_blocks=8, block_size=4, n_layer=1, n_heads=2, head_dim=4)
    req = PagedRequest(request_id=0, pool=pool)

    # Simulate writing 6 tokens' worth of fake K/V for layer 0
    for i in range(6):
        req.append_token_slot()
        fake_k = torch.full((2, 4), float(i))   # distinct value per token, so we can verify order
        fake_v = torch.full((2, 4), float(i) * 10)
        write_kv(pool, req, layer_idx=0, k=fake_k, v=fake_v)

    k_full, v_full = gather_kv(pool, req, layer_idx=0)
    print("k_full shape:", k_full.shape)
    print("k_full[:, 0, 0] (should read 0,1,2,3,4,5):", k_full[:, 0, 0])
    print("v_full[:, 0, 0] (should read 0,10,20,30,40,50):", v_full[:, 0, 0])