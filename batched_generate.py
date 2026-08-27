import torch
from model import tokenizer, wte, wpe, blocks, ln_f_w, ln_f_b, layer_norm
import math

# GPT-2's tokenizer has no pad token by default — reuse eos_token as pad
tokenizer.pad_token = tokenizer.eos_token

prompts = [
    "Hello",
    "The weather today is",
    "I really enjoy programming because"
]

# Left-padding, return attention mask too
tokenizer.padding_side = "left"
encoded = tokenizer(prompts, return_tensors="pt", padding=True)

input_ids = encoded["input_ids"]
attention_mask = encoded["attention_mask"]

print("input_ids shape:", input_ids.shape)
print("input_ids:\n", input_ids)
print("\nattention_mask:\n", attention_mask)


def split_heads(t, n_heads=12, head_dim=64):
    B, T, C = t.shape
    return t.view(B, T, n_heads, head_dim).transpose(1, 2)


def attention_batched(x, block, attn_mask):
    B, T, C = x.shape
    n_heads, head_dim = 12, 64

    qkv = x @ block["attn_w"] + block["attn_b"]
    q, k, v = qkv.split(C, dim=-1)

    q = split_heads(q, n_heads, head_dim)
    k = split_heads(k, n_heads, head_dim)
    v = split_heads(v, n_heads, head_dim)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)  # [B, 12, T, T]

    causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))

   
    pad_mask = attn_mask[:, None, None, :].bool()
    self_mask = torch.eye(T, dtype=torch.bool).unsqueeze(0).unsqueeze(0)

    combined_mask = (causal_mask.unsqueeze(0).unsqueeze(0) & pad_mask) | self_mask  

    scores = scores.masked_fill(~combined_mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = probs @ v
    out = out.transpose(1, 2).contiguous().view(B, T, C)

    out = out @ block["attn_proj_w"] + block["attn_proj_b"]
    return out

def transformer_block_batched(x, block, attn_mask):
    normed = layer_norm(x, block["ln1_w"], block["ln1_b"])
    x = x + attention_batched(normed, block, attn_mask)

    normed2 = layer_norm(x, block["ln2_w"], block["ln2_b"])
    x = x + mlp(normed2, block)

    return x


def mlp(x, block):
    h = x @ block["mlp_fc_w"] + block["mlp_fc_b"]
    h = 0.5 * h * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (h + 0.044715 * h.pow(3.0))))
    h = h @ block["mlp_proj_w"] + block["mlp_proj_b"]
    return h


def forward_batched(input_ids, attn_mask):
    B, T = input_ids.shape
    position_ids = attn_mask.cumsum(dim=-1) - 1
    position_ids = position_ids.clamp(min=0)

    x = wte[input_ids] + wpe[position_ids]

    for block in blocks:
        x = transformer_block_batched(x, block, attn_mask)

    x = layer_norm(x, ln_f_w, ln_f_b)
    logits = x @ wte.T
    return logits

@torch.no_grad()
def generate_batched(prompts, max_new_tokens=20):
    tokenizer.padding_side = "left"
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"]
    attn_mask = encoded["attention_mask"]

    for _ in range(max_new_tokens):
        logits = forward_batched(input_ids, attn_mask)
        next_ids = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)  # [B, 1]

        input_ids = torch.cat([input_ids, next_ids], dim=1)
        attn_mask = torch.cat([attn_mask, torch.ones_like(next_ids)], dim=1)

    return [tokenizer.decode(seq, skip_special_tokens=True) for seq in input_ids]


if __name__ == "__main__":
    prompts = [
        "Hello",
        "The weather today is",
        "I really enjoy programming because"
    ]
    outputs = generate_batched(prompts, max_new_tokens=20)
    for p, o in zip(prompts, outputs):
        print(f"Prompt: {p!r}\nOutput: {o!r}\n")