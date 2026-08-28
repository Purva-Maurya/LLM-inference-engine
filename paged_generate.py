import math
import torch
from model import tokenizer, wte, wpe, blocks, ln_f_w, ln_f_b, layer_norm, mlp, split_heads
from paged_cache import BlockPool, PagedRequest, write_kv, gather_kv


def attention_paged(x, block, layer_idx, pool, req):
    B, T, C = x.shape  # B=1, T=1 (one new token at a time)
    n_heads, head_dim = 12, 64

    qkv = x @ block["attn_w"] + block["attn_b"]
    q, k, v = qkv.split(C, dim=-1)

    q = split_heads(q, n_heads, head_dim)              # [1, 12, 1, 64]
    k_new = split_heads(k, n_heads, head_dim)[0, :, 0, :]  # [12, 64]
    v_new = split_heads(v, n_heads, head_dim)[0, :, 0, :]

    write_kv(pool, req, layer_idx, k_new, v_new)
    k_full, v_full = gather_kv(pool, req, layer_idx)   # [seq_len, 12, 64] each

    k_full = k_full.permute(1, 0, 2).unsqueeze(0)       # [1, 12, seq_len, 64]
    v_full = v_full.permute(1, 0, 2).unsqueeze(0)

    scores = (q @ k_full.transpose(-2, -1)) / math.sqrt(head_dim)  # [1, 12, 1, seq_len]
    probs = torch.softmax(scores, dim=-1)
    out = probs @ v_full                                # [1, 12, 1, 64]
    out = out.transpose(1, 2).contiguous().view(B, T, C)

    out = out @ block["attn_proj_w"] + block["attn_proj_b"]
    return out


def transformer_block_paged(x, block, layer_idx, pool, req):
    normed = layer_norm(x, block["ln1_w"], block["ln1_b"])
    x = x + attention_paged(normed, block, layer_idx, pool, req)

    normed2 = layer_norm(x, block["ln2_w"], block["ln2_b"])
    x = x + mlp(normed2, block)

    return x

def forward_paged(token_id, position, pool, req):
    input_ids = torch.tensor([[token_id]])
    x = wte[input_ids] + wpe[torch.tensor([[position]])]

    for layer_idx, block in enumerate(blocks):
        x = transformer_block_paged(x, block, layer_idx, pool, req)

    x = layer_norm(x, ln_f_w, ln_f_b)
    logits = x @ wte.T
    return logits


@torch.no_grad()
def generate_paged(prompt, max_new_tokens, pool):
    prompt_ids = tokenizer.encode(prompt)
    req = PagedRequest(request_id=0, pool=pool)

    logits = None
    # Prefill: feed prompt tokens one at a time (simple, if not maximally efficient)
    for pos, tid in enumerate(prompt_ids):
        req.append_token_slot()
        logits = forward_paged(tid, pos, pool, req)

    next_id = torch.argmax(logits[0, -1, :]).item()
    generated = prompt_ids + [next_id]

    # Decode: one new token at a time
    for _ in range(max_new_tokens - 1):
        pos = req.num_tokens
        req.append_token_slot()
        logits = forward_paged(next_id, pos, pool, req)

        next_id = torch.argmax(logits[0, -1, :]).item()
        generated.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

    req.free()
    return tokenizer.decode(generated)


if __name__ == "__main__":
    pool = BlockPool(num_blocks=64, block_size=4)
    output = generate_paged("The future of artificial intelligence is", max_new_tokens=30, pool=pool)
    print(output)
