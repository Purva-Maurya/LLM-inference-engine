import math
import torch
from model import tokenizer, wte, wpe, blocks, ln_f_w, ln_f_b, layer_norm, split_heads,mlp

def attention_cached(x, block, cache=None):
    B, T, C = x.shape
    n_heads, head_dim = 12, 64

    qkv = x @ block["attn_w"] + block["attn_b"]
    q, k, v = qkv.split(C, dim=-1)

    q = split_heads(q, n_heads, head_dim)
    k = split_heads(k, n_heads, head_dim)
    v = split_heads(v, n_heads, head_dim)

    past_len = 0
    if cache is not None:
        past_k, past_v = cache
        past_len = past_k.shape[2]
        k = torch.cat([past_k, k], dim=2)
        v = torch.cat([past_v, v], dim=2)

    new_cache = (k, v)
    total_len = k.shape[2]

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)  # [1, 12, T, total_len]

    # Causal mask using ABSOLUTE positions: query i (at position past_len+i)
    # may attend to key j (at position j) only if j <= past_len+i.
    query_positions = torch.arange(past_len, past_len + T).unsqueeze(1)   # [T, 1]
    key_positions = torch.arange(0, total_len).unsqueeze(0)               # [1, total_len]
    causal_mask = key_positions <= query_positions                        # [T, total_len]

    scores = scores.masked_fill(~causal_mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = probs @ v
    out = out.transpose(1, 2).contiguous().view(B, T, C)

    out = out @ block["attn_proj_w"] + block["attn_proj_b"]
    return out, new_cache

def transformer_block_cached(x, block, cache=None):
    normed = layer_norm(x, block["ln1_w"], block["ln1_b"])
    attn_out, new_cache = attention_cached(normed, block, cache)
    x = x + attn_out

    normed2 = layer_norm(x, block["ln2_w"], block["ln2_b"])
    x = x + mlp(normed2, block)

    return x, new_cache

def forward_cached(input_ids, past_caches=None, past_length=0):
    seq_len = input_ids.shape[1]  # will be 1 after the first step

    position_ids = torch.arange(past_length, past_length + seq_len)
    x = wte[input_ids] + wpe[position_ids]

    if past_caches is None:
        past_caches = [None] * len(blocks)

    new_caches = []
    for block, cache in zip(blocks, past_caches):
        x, new_cache = transformer_block_cached(x, block, cache)
        new_caches.append(new_cache)

    x = layer_norm(x, ln_f_w, ln_f_b)
    logits = x @ wte.T
    return logits, new_caches

from sampling import sample_with_temperature_and_topp

from sampling import sample_with_temperature_and_topp

@torch.no_grad()
def generate_cached(prompt, max_new_tokens=30, temperature=1.0, top_p=None):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    # --- Prefill: process the whole prompt at once ---
    logits, caches = forward_cached(input_ids, past_caches=None, past_length=0)
    past_length = input_ids.shape[1]

    if top_p is not None:
        next_id = sample_with_temperature_and_topp(logits[0, -1, :], temperature, top_p)
    else:
        next_id = torch.argmax(logits[0, -1, :]).item()
    generated = input_ids[0].tolist() + [next_id]

    # --- Decode: one new token at a time, reusing the cache ---
    for _ in range(max_new_tokens - 1):
        next_input = torch.tensor([[next_id]])
        logits, caches = forward_cached(next_input, past_caches=caches, past_length=past_length)
        past_length += 1

        if top_p is not None:
            next_id = sample_with_temperature_and_topp(logits[0, -1, :], temperature, top_p)
        else:
            next_id = torch.argmax(logits[0, -1, :]).item()
        generated.append(next_id)

        if next_id == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated)

if __name__ == "__main__":
    output = generate_cached("The future of artificial intelligence is", max_new_tokens=30)
    print(output)

