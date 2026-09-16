import math
import asyncio
import torch
from model import tokenizer, wte, wpe, blocks, ln_f_w, ln_f_b, layer_norm, mlp, split_heads
from paged_cache import PagedRequest, write_kv, gather_kv
from paged_generate import forward_paged  # reused as-is, for prefill only
from sampling import sample_with_temperature_and_topp

N_HEADS, HEAD_DIM = 12, 64


class ScheduledRequest:
    """One in-flight request: its own paged KV state, generation progress,
    and the asyncio Future the HTTP handler awaits for the final text."""
    def __init__(self, request_id, prompt, pool, max_new_tokens=30, temperature=1.0, top_p=0.9):
        self.request_id = request_id
        self.prompt_ids = tokenizer.encode(prompt)
        self.paged_req = PagedRequest(request_id=request_id, pool=pool)
        self.pool = pool
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        self.generated_ids = list(self.prompt_ids)
        self.num_generated = 0
        self.next_token_id = None
        self.next_position = 0
        self.future = asyncio.get_event_loop().create_future()

    def is_done(self):
        if self.num_generated >= self.max_new_tokens:
            return True
        if self.generated_ids and self.generated_ids[-1] == tokenizer.eos_token_id:
            return True
        return False

    def finish(self):
        text = tokenizer.decode(self.generated_ids, skip_special_tokens=True)
        if not self.future.done():
            self.future.set_result(text)


def _sample(req, last_logits):
    if req.top_p is not None:
        return sample_with_temperature_and_topp(last_logits, req.temperature, req.top_p)
    return torch.argmax(last_logits).item()


def _prefill(req: ScheduledRequest):
    """Feed the whole prompt through the paged cache, one token at a time —
    reuses forward_paged from paged_generate.py exactly as-is. Runs once,
    at admission, before the request joins the batched decode loop."""
    logits = None
    for pos, tid in enumerate(req.prompt_ids):
        req.paged_req.append_token_slot()
        logits = forward_paged(tid, pos, req.pool, req.paged_req)

    next_id = _sample(req, logits[0, -1, :])
    req.generated_ids.append(next_id)
    req.num_generated += 1
    req.next_token_id = next_id
    req.next_position = len(req.prompt_ids)


def _attention_paged_batched(x, block, layer_idx, active):
    """Batches the shared matmuls across all active requests' single new
    tokens, while reading/writing each request's OWN paged KV blocks
    individually (different requests have unrelated block tables and
    different history lengths)."""
    B = len(active)
    C = x.shape[-1]

    qkv = x @ block["attn_w"] + block["attn_b"]  # [B,1,3C] — one shared matmul
    q, k, v = qkv.split(C, dim=-1)

    q = split_heads(q, N_HEADS, HEAD_DIM)                       # [B,12,1,64]
    k_new = split_heads(k, N_HEADS, HEAD_DIM)[:, :, 0, :]       # [B,12,64]
    v_new = split_heads(v, N_HEADS, HEAD_DIM)[:, :, 0, :]

    k_list, v_list, lens = [], [], []
    for i, r in enumerate(active):
        write_kv(r.pool, r.paged_req, layer_idx, k_new[i], v_new[i])
        k_full, v_full = gather_kv(r.pool, r.paged_req, layer_idx)
        k_list.append(k_full)
        v_list.append(v_full)
        lens.append(k_full.shape[0])

    max_len = max(lens)
    k_batch = torch.zeros(B, max_len, N_HEADS, HEAD_DIM)
    v_batch = torch.zeros(B, max_len, N_HEADS, HEAD_DIM)
    pad_mask = torch.zeros(B, max_len, dtype=torch.bool)
    for i in range(B):
        L = lens[i]
        k_batch[i, :L] = k_list[i]
        v_batch[i, :L] = v_list[i]
        pad_mask[i, :L] = True

    k_batch = k_batch.permute(0, 2, 1, 3)  # [B,12,max_len,64]
    v_batch = v_batch.permute(0, 2, 1, 3)

    scores = (q @ k_batch.transpose(-2, -1)) / math.sqrt(HEAD_DIM)  # [B,12,1,max_len]
    scores = scores.masked_fill(~pad_mask[:, None, None, :], float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = probs @ v_batch                                        # [B,12,1,64]
    out = out.transpose(1, 2).contiguous().view(B, 1, C)
    return out @ block["attn_proj_w"] + block["attn_proj_b"]


def _transformer_block_paged_batched(x, block, layer_idx, active):
    normed = layer_norm(x, block["ln1_w"], block["ln1_b"])
    x = x + _attention_paged_batched(normed, block, layer_idx, active)
    normed2 = layer_norm(x, block["ln2_w"], block["ln2_b"])
    x = x + mlp(normed2, block)
    return x


@torch.no_grad()
def decode_step_batched(active):
    """One batched decode step: every active request contributes exactly
    one new token, processed together through all 12 layers."""
    token_ids = torch.tensor([[r.next_token_id] for r in active])
    positions = torch.tensor([[r.next_position] for r in active])
    x = wte[token_ids] + wpe[positions]

    for layer_idx, block in enumerate(blocks):
        x = _transformer_block_paged_batched(x, block, layer_idx, active)

    x = layer_norm(x, ln_f_w, ln_f_b)
    return x @ wte.T


class PagedScheduler:
    def __init__(self, pool, max_batch_size=4, step_delay=0.0):
        self.pool = pool
        self.max_batch_size = max_batch_size
        self.step_delay = step_delay
        self.waiting = []
        self.active = []

    def submit(self, req: ScheduledRequest):
        self.waiting.append(req)

    async def run_forever(self):
        while True:
            while len(self.active) < self.max_batch_size and self.waiting:
                new_req = self.waiting.pop(0)
                await asyncio.get_event_loop().run_in_executor(None, _prefill, new_req)
                self.active.append(new_req)

            if not self.active:
                await asyncio.sleep(0.01)
                continue

            self._decode_step()
            await asyncio.sleep(self.step_delay)

    def _decode_step(self):
        for r in self.active:
            r.paged_req.append_token_slot()

        logits = decode_step_batched(self.active)

        still_active = []
        for i, r in enumerate(self.active):
            next_id = _sample(r, logits[i, -1, :])
            r.generated_ids.append(next_id)
            r.num_generated += 1
            r.next_token_id = next_id
            r.next_position += 1

            if r.is_done():
                r.finish()
                r.paged_req.free()
            else:
                still_active.append(r)
        self.active = still_active