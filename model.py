import math
import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# ---- Load model and tokenizer ----
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
hf_model = GPT2LMHeadModel.from_pretrained("gpt2")
sd = hf_model.state_dict()

wte = sd["transformer.wte.weight"]
wpe = sd["transformer.wpe.weight"]
ln_f_w = sd["transformer.ln_f.weight"]
ln_f_b = sd["transformer.ln_f.bias"]

n_layer = 12
n_heads = 12
head_dim = 64

blocks = []
for i in range(n_layer):
    p = f"transformer.h.{i}."
    block = {
        "ln1_w": sd[p + "ln_1.weight"], "ln1_b": sd[p + "ln_1.bias"],
        "attn_w": sd[p + "attn.c_attn.weight"], "attn_b": sd[p + "attn.c_attn.bias"],
        "attn_proj_w": sd[p + "attn.c_proj.weight"], "attn_proj_b": sd[p + "attn.c_proj.bias"],
        "ln2_w": sd[p + "ln_2.weight"], "ln2_b": sd[p + "ln_2.bias"],
        "mlp_fc_w": sd[p + "mlp.c_fc.weight"], "mlp_fc_b": sd[p + "mlp.c_fc.bias"],
        "mlp_proj_w": sd[p + "mlp.c_proj.weight"], "mlp_proj_b": sd[p + "mlp.c_proj.bias"],
    }
    blocks.append(block)


# ---- Core building blocks ----
def layer_norm(x, weight, bias, eps=1e-5):
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm * weight + bias


def split_heads(t, n_heads=12, head_dim=64):
    B, T, C = t.shape
    t = t.view(B, T, n_heads, head_dim)
    t = t.transpose(1, 2)
    return t


def attention(x, block):
    B, T, C = x.shape
    qkv = x @ block["attn_w"] + block["attn_b"]
    q, k, v = qkv.split(C, dim=-1)

    q = split_heads(q, n_heads, head_dim)
    k = split_heads(k, n_heads, head_dim)
    v = split_heads(v, n_heads, head_dim)

    scores = (q @ k.transpose(-2, -1)) / math.sqrt(head_dim)

    causal_mask = torch.tril(torch.ones(T, T, dtype=torch.bool))
    scores = scores.masked_fill(~causal_mask, float("-inf"))

    probs = torch.softmax(scores, dim=-1)
    out = probs @ v
    out = out.transpose(1, 2).contiguous().view(B, T, C)

    out = out @ block["attn_proj_w"] + block["attn_proj_b"]
    return out


def mlp(x, block):
    h = x @ block["mlp_fc_w"] + block["mlp_fc_b"]
    h = 0.5 * h * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (h + 0.044715 * h.pow(3.0))))
    h = h @ block["mlp_proj_w"] + block["mlp_proj_b"]
    return h


def transformer_block(x, block):
    normed = layer_norm(x, block["ln1_w"], block["ln1_b"])
    x = x + attention(normed, block)

    normed2 = layer_norm(x, block["ln2_w"], block["ln2_b"])
    x = x + mlp(normed2, block)

    return x


def forward(input_ids):
    seq_len = input_ids.shape[1]
    x = wte[input_ids] + wpe[torch.arange(seq_len)]

    for block in blocks:
        x = transformer_block(x, block)

    x = layer_norm(x, ln_f_w, ln_f_b)
    logits = x @ wte.T
    return logits


def greedy_generate(prompt, max_new_tokens=20):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = forward(input_ids)
            next_id = torch.argmax(logits[0, -1, :]).item()
            next_id_tensor = torch.tensor([[next_id]])
            input_ids = torch.cat([input_ids, next_id_tensor], dim=1)

            if next_id == tokenizer.eos_token_id:
                break

    return tokenizer.decode(input_ids[0])


if __name__ == "__main__":
    output = greedy_generate("The future of artificial intelligence is", max_new_tokens=30)
    print(output)
